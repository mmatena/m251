"""TODO: Add title."""
import contextlib
import re

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
    num_examples,
    finetuned_ckpt_uuid=None,
    fisher_class_chunk_size=4096,
    y_samples=None,
):
    if finetuned_ckpt_uuid is None:
        ctx = contextlib.suppress()
    else:
        ctx = scopes.binding_by_name_scope("checkpoint", finetuned_ckpt_uuid)
    with ctx:
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


# TODO: Move this to common place.
def _is_uuid(s):
    return re.match(r"^[0-9a-f]{32}$", s)


@executable.executable(
    default_bindings={
        "initializer": gc_exe.bert_initializer,
        "loader": gc_exe.bert_loader,
        "builder": gc_exe.bert_builder,
    },
)
def diagonal_mergeable_model_from_checkpoint_or_pretrained(
    checkpoint,
    checkpoint_to_fisher_matrix_uuid,
    pretrained_model,
    _initializer,
    _builder,
    _loader,
    storage,
):
    with tf.device("/cpu"):
        if checkpoint is None or _is_uuid(checkpoint):
            bindings = [("checkpoint", checkpoint)]
        else:
            bindings = [("pretrained_model", checkpoint), ("checkpoint", None)]

        with scopes.binding_by_name_scopes(bindings):
            ft_model = _initializer()
            with scopes.binding_by_name_scope("model", ft_model):
                ft_model = _builder(ft_model)
                ft_model = _loader(ft_model)

        if checkpoint is not None:
            fisher_matrix_uuid = checkpoint_to_fisher_matrix_uuid[checkpoint]
        else:
            fisher_matrix_uuid = checkpoint_to_fisher_matrix_uuid[pretrained_model]

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
    normalize_fishers=False,
    multitask_merge=False,
):
    to_be_merged = _initializer()
    with scopes.binding_by_name_scope("model", to_be_merged):
        to_be_merged = _builder(to_be_merged)

        merged_models = diagonal.merge_models_with_weightings(
            to_be_merged,
            mergeable_models,
            weightings,
            single_task=not multitask_merge,
            min_fisher=min_fisher,
            normalize_fishers=normalize_fishers,
        )

        for merged in merged_models:
            compile_kwargs = {}
            if _metrics:
                compile_kwargs["metrics"] = _metrics(merged)

            merged.compile(**compile_kwargs)

            yield merged


###############################################################################


@executable.executable(
    default_bindings={
        "initializer": gc_exe.bert_initializer,
        "builder": gc_exe.bert_builder,
    },
)
def diagonal_model_merge_weighting_search(
    mergeable_models,
    merge_weighting_search_steps,
    merge_weighting_num_inits,
    _initializer,
    _builder,
    _model_scorer,
    min_fisher=1e-6,
    multitask_merge=False,
    merge_on_cpu=False,
):
    to_be_merged = _initializer()
    with scopes.binding_by_name_scope("model", to_be_merged):
        to_be_merged = _builder(to_be_merged)

        (
            merged_model,
            weighting,
            trial_weightings,
            trial_scores,
        ) = diagonal.merge_search_best_weighting(
            to_be_merged,
            mergeable_models=mergeable_models,
            score_fn=_model_scorer,
            max_evals=merge_weighting_search_steps,
            num_inits=merge_weighting_num_inits,
            min_fisher=min_fisher,
            single_task=not multitask_merge,
            merge_on_cpu=merge_on_cpu,
        )

    merged_model.compile()

    return merged_model, weighting, trial_weightings, trial_scores


###############################################################################


@executable.executable()
def diagonal_regularize_ewc_from_initial(
    model,
    fisher_matrix_uuid,
    storage,
    reg_strength=0.0,
):
    assert not model.is_hf
    if not reg_strength:
        return model

    logging.info(f"Retrieving saved fisher matrix: {fisher_matrix_uuid}")
    with storage.retrieve_blob_as_tempfile(fisher_matrix_uuid) as f:
        logging.info(f"Loading retrieved fisher matrix: {fisher_matrix_uuid}")
        fisher_matrix = diagonal.DiagonalFisherMatrix.load(f.name)

    diags = fisher_matrix.fisher_diagonals
    og_weights = [tf.identity(w) for w in model.get_mergeable_variables()]

    def regularizer(model_during_training):
        trainable_weights = model_during_training.get_mergeable_variables()
        from_pt_l2 = [
            tf.reduce_sum(diag * tf.square(w - og_w))
            for diag, og_w, w in zip(diags, og_weights, trainable_weights)
        ]
        return reg_strength * tf.reduce_sum(from_pt_l2)

    model.add_regularizer(regularizer)

    return model
