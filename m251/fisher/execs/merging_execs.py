"""TODO: Add title."""
from absl import logging
import tensorflow as tf

from del8.core import data_class
from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.evaluation import eval_execs


@data_class.data_class()
class MergingEvaluationResults(object):
    def __init__(
        self,
        results,
        checkpoints,
        tasks,
        weighting,
        trial_weightings=None,
        trial_scores=None,
    ):
        pass


@executable.executable()
def merging_evaluation_results_saver(
    results,
    storage,
    checkpoints,
    tasks,
    weighting,
    trial_weightings=None,
    trial_scores=None,
):
    item = MergingEvaluationResults(
        results,
        checkpoints=checkpoints,
        tasks=tasks,
        weighting=weighting,
        trial_weightings=trial_weightings,
        trial_scores=trial_scores,
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
    multitask_merge=False,
    additional_model_bindings=None,
):
    assert len(checkpoints) == len(tasks)
    additional_model_bindings = additional_model_bindings or len(tasks) * [[]]

    mergeable_models = []
    for checkpoint, task, extra_bindings in zip(
        checkpoints, tasks, additional_model_bindings
    ):
        bindings = [
            ("checkpoint", checkpoint),
            ("tasks", [task]),
            ("task", task),
        ]
        bindings.extend(extra_bindings)
        with scopes.binding_by_name_scopes(bindings):
            mergeable_model = _mergeable_model()
            mergeable_models.append(mergeable_model)

    # NOTE: Single task merge means we only care about the performance of
    # the first task.
    bindings = [
        ("tasks", tasks if multitask_merge else tasks[:1]),
        ("task", tasks[0]),
    ]
    with scopes.binding_by_name_scopes(bindings):
        merged_models = _model_merger(
            mergeable_models=mergeable_models, weightings=weightings
        )
        for merged_model, weighting in zip(merged_models, weightings):
            logging.info(f"Evaluating task weighting {weighting}")
            with scopes.binding_by_name_scope("weighting", weighting):
                _evaluate_model(merged_model)


###############################################################################


@executable.executable()
def pass_evaluation_results(results):
    return results


def _average_dict_values(d):
    values = d.values()
    return sum(values) / len(values)


@executable.executable()
def single_score_from_results(results, task):
    if task == "mnli":
        scores = [
            results.get("mnli", None),
            results.get("mnli_mismatched", None),
            results.get("mnli_mismatched", None),
        ]
        scores = [s for s in scores if s is not None]
        assert scores, "No keys found for mnli."
    else:
        scores = [results[task]]

    scores = [_average_dict_values(s) for s in scores]

    return sum(scores) / len(scores)


def _average_score(scores):
    if not isinstance(scores, dict):
        return scores
    values = [_average_score(v) for v in scores.values()]
    return sum(values) / len(values)


@executable.executable()
def average_score_from_results(results):
    return _average_score(results)


@executable.executable(
    default_bindings={
        "evaluate_model": eval_execs.robust_evaluate_model,
        "single_score_from_results": single_score_from_results,
    }
)
def merge_weighting_search_scorer(
    merged_model,
    _evaluate_model,
    _single_score_from_results,
):
    merged_model.compile()
    # The evaluate_model returns the result of this function, so make it
    # the identity function.
    with scopes.binding_by_name_scope(
        "evaluation_results_saver", pass_evaluation_results
    ):
        results = _evaluate_model(merged_model)
    return _single_score_from_results(results)


@executable.executable(
    default_bindings={
        "evaluate_model": eval_execs.robust_evaluate_model,
        "model_scorer": merge_weighting_search_scorer,
        "evaluation_results_saver": merging_evaluation_results_saver,
    },
)
def merge_weighting_search_from_checkpoints(
    tasks,
    #
    checkpoints,
    checkpoint_tasks,
    #
    search_num_examples,
    final_evaluate_num_examples,
    #
    _mergeable_model,
    _merge_weighting_searcher,
    #
    _evaluate_model,
):
    mergeable_models = []
    assert len(checkpoints) == len(checkpoint_tasks)
    for checkpoint, t in zip(checkpoints, checkpoint_tasks):
        bindings = [
            ("checkpoint", checkpoint),
            ("task", t),
            ("tasks", [t]),
        ]
        with scopes.binding_by_name_scopes(bindings):
            mergeable_model = _mergeable_model()
            mergeable_models.append(mergeable_model)

    bindings = [
        ("num_examples", search_num_examples),
        ("tasks", tasks),
    ]
    with scopes.binding_by_name_scopes(bindings):
        (
            merged_model,
            weighting,
            trial_weightings,
            trial_scores,
        ) = _merge_weighting_searcher(mergeable_models)

    bindings = [
        ("weighting", weighting),
        ("trial_weightings", trial_weightings),
        ("trial_scores", trial_scores),
        ("num_examples", final_evaluate_num_examples),
        ("tasks", tasks),
    ]
    with scopes.binding_by_name_scopes(bindings):
        _evaluate_model(merged_model)
