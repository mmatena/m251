"""TODO: Add title."""
from absl import logging
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes
from del8.executables.models import checkpoints as ckpt_exec

from m251.models.bert import glue_classifier_execs as gc_exe

from . import diagonal


@executable.executable(
    default_bindings={
        "initializer": gc_exe.bert_initializer,
        "loader": ckpt_exec.checkpoint_loader,
        "builder": gc_exe.bert_builder,
    }
)
def diagonal_fisher_computer(
    _initializer,
    _builder,
    _loader,
    finetuned_ckpt_uuid,
    num_examples,
    fisher_class_chunk_size=4096,
    y_samples=None,
):
    with scopes.binding_by_name_scope("checkpoint", finetuned_ckpt_uuid):
        ft_model = _initializer()
        with scopes.binding_by_name_scope("model", ft_model):
            ft_model = _builder(ft_model)
            ft_model = _loader(ft_model)

        computer = diagonal.DiagonalFisherComputer(
            ft_model,
            total_examples=num_examples,
            y_samples=y_samples,
            class_chunk_size=fisher_class_chunk_size,
        )
        # NOTE: I don't really think the next binding will ever be used, but I'm
        # putting here out of paranoia.
        with scopes.binding_by_name_scope("model", computer):
            computer.compile()

    return computer


###############################################################################


class MergableModel(object):
    def __init__(self, model, fisher_matrix):
        self.model = model
        self.fisher_matrix = fisher_matrix


@executable.executable(
    default_bindings={
        "initializer": gc_exe.bert_initializer,
        "loader": ckpt_exec.checkpoint_loader,
        "builder": gc_exe.bert_builder,
    },
)
def diagonal_mergeable_model_from_checkpoint(
    checkpoint,
    checkpoint_to_fisher_matrix_uuid,
    _initializer,
    _builder,
    _loader,
    storage,
):
    with tf.device("/cpu"):
        with scopes.binding_by_name_scope("checkpoint", checkpoint):
            ft_model = _initializer()
            with scopes.binding_by_name_scope("model", ft_model):
                ft_model = _builder(ft_model)
                ft_model = _loader(ft_model)

        fisher_matrix_uuid = checkpoint_to_fisher_matrix_uuid[checkpoint]
        logging.info(f"Retrieving saved fisher matrix: {fisher_matrix_uuid}")
        with storage.retrieve_blob_as_tempfile(fisher_matrix_uuid) as f:
            logging.info(f"Loading retrieved fisher matrix: {fisher_matrix_uuid}")
            fisher_matrix = diagonal.DiagonalFisherMatrix.load(f.name)

        return MergableModel(model=ft_model, fisher_matrix=fisher_matrix)


###############################################################################


@executable.executable(
    default_bindings={
        "initializer": gc_exe.bert_initializer,
        "builder": gc_exe.bert_builder,
        "metrics": gc_exe.glue_finetuning_metrics,
    },
)
def diagonal_model_merger(
    mergeable_models,
    weightings,
    _initializer,
    _builder,
    _metrics=None,
    min_fisher=1e-6,
):
    to_be_merged = _initializer()
    with scopes.binding_by_name_scope("model", to_be_merged):
        to_be_merged = _builder(to_be_merged)

        with tf.device("/cpu"):
            merged_models = diagonal.merge_models_with_weightings(
                to_be_merged, mergeable_models, weightings, min_fisher=min_fisher
            )

        for merged in merged_models:
            compile_kwargs = {}
            if _metrics:
                compile_kwargs["metrics"] = _metrics(merged)

            merged.compile(**compile_kwargs)

            yield merged
