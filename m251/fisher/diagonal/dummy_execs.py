"""TODO: Add title."""
import contextlib

from absl import logging
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes
from del8.executables.models import checkpoints as ckpt_exec

from m251.models.bert import glue_classifier_execs as gc_exe
from . import diagonal_execs


def dummy_merge_models(merged_model, mergeable_models, weighting, single_task=True):
    assert len(mergeable_models) == len(weighting)

    with tf.device("gpu"):
        denom = tf.reduce_sum(weighting)
        for i, var in enumerate(merged_model.get_mergeable_variables()):
            rhs = []
            for j, (weight, mm) in enumerate(zip(weighting, mergeable_models)):
                model = mm.model
                rhs.append(weight * model.get_mergeable_variables()[i])
            rhs = tf.reduce_sum(rhs, axis=0)
            var.assign(rhs / denom)

    if single_task:
        heads = [mergeable_models[0].model.get_classifier_head()]
    else:
        heads = [m.model.get_classifier_head() for m in mergeable_models]
    merged_model.set_classifier_heads(heads)

    return merged_model


@executable.executable(
    default_bindings={
        "initializer": gc_exe.bert_initializer,
        "builder": gc_exe.bert_builder,
        "metrics": gc_exe.glue_finetuning_metrics,
    },
)
def dummy_fisher_model_merger(
    mergeable_models,
    weightings,
    _initializer,
    _builder,
    _metrics=None,
    multitask_merge=False,
):
    to_be_merged = _initializer()
    with scopes.binding_by_name_scope("model", to_be_merged):
        to_be_merged = _builder(to_be_merged)

        for weighting in weightings:
            merged = dummy_merge_models(
                to_be_merged,
                mergeable_models,
                weighting,
                single_task=not multitask_merge,
            )
            compile_kwargs = {}
            if _metrics:
                compile_kwargs["metrics"] = _metrics(merged)

            merged.compile(**compile_kwargs)

            logging.info("DUMMY MERGING!!!")

            yield merged


###############################################################################


@executable.executable(
    default_bindings={
        "diagonal_mergeable_model_from_checkpoint": diagonal_execs.diagonal_mergeable_model_from_checkpoint
    },
)
def shuffled_mergeable_model_from_checkpoint(_diagonal_mergeable_model_from_checkpoint):
    mergeable_model = _diagonal_mergeable_model_from_checkpoint()
    for d in mergeable_model.fisher_matrix.get_diagonals():
        shape = tf.shape(d)
        shuffled = tf.reshape(tf.random.shuffle(tf.reshape(d, [-1])), shape)
        d.assign(shuffled)
    return mergeable_model
