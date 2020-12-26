"""TODO: Add title."""
import tensorflow as tf

from del8.core import data_class
from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.evaluation import eval_execs


@data_class.data_class()
class MergingEvaluationResults(object):
    def __init__(self, results, checkpoints, tasks, weighting):
        pass


@executable.executable()
def merging_evaluation_results_saver(results, storage, checkpoints, tasks, weighting):
    item = MergingEvaluationResults(
        results,
        checkpoints=checkpoints,
        tasks=tasks,
        weighting=weighting,
    )
    item_uuid = storage.store_item(item)
    return item_uuid


@executable.executable(
    default_bindings={
        "evaluate_model": eval_execs.evaluate_model,
        "evaluation_results_saver": merging_evaluation_results_saver,
    },
)
def merge_and_evaluate_from_checkpoints(
    checkpoints,
    tasks,
    # TODO: Describe weightings, length is independent of checkpoints and tasks.
    weightings,
    _mergeable_model,
    _model_merger,
    _evaluate_model,
    storage,
):
    assert len(checkpoints) == len(tasks)
    with storage.blob_read_cache():
        # NOTE: I don't think we can clear a part of the graph with tf, so
        # we have to create all the models from files to prevent OOMs. We
        # cache reads from the blob storage, so this will hopefully be
        # reasonably performant.
        for weighting in weightings:
            mergeable_models = []
            for checkpoint, task in zip(checkpoints, tasks):
                bindings = [
                    ("checkpoint", checkpoint),
                    ("tasks", [task]),
                    ("task", task),
                ]
                with scopes.binding_by_name_scopes(bindings):
                    mergeable_model = _mergeable_model()
                    mergeable_models.append(mergeable_model)

            with scopes.binding_by_name_scope("weighting", weighting):
                # NOTE: merged_model should be compiled and ready to eval.
                merged_model = _model_merger(
                    mergeable_models=mergeable_models, weighting=weighting
                )
                _evaluate_model(merged_model)

                tf.keras.backend.clear_session()
