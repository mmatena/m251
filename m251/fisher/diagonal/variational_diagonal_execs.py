"""TODO: Add title."""
import contextlib

from absl import logging
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes
from del8.executables.models import checkpoints as ckpt_exec

from m251.models.bert import glue_classifier_execs as gc_exe

from .diagonal_execs import MergableModel
from . import variational_diagonal as vardiag


@executable.executable(
    default_bindings={
        "initializer": gc_exe.bert_initializer,
        "loader": ckpt_exec.checkpoint_loader,
        "builder": gc_exe.bert_builder,
    }
)
def variational_diag_fisher_computer(
    _initializer,
    _builder,
    _loader,
    optimizer,
    finetuned_ckpt_uuid=None,
    variational_fisher_beta=1e-8,
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

        computer = vardiag.VariationalDiagFisherComputer(
            ft_model,
            beta=variational_fisher_beta,
        )
        # NOTE: I don't really think the next binding will ever be used, but I'm
        # putting here out of paranoia.
        with scopes.binding_by_name_scope("model", computer):
            computer.compile(optimizer=optimizer)

    return computer


###############################################################################


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
            fisher_matrix = vardiag.VariationalDiagFisherMatrix.load(
                f.name, take_exp=True
            )

        return MergableModel(model=ft_model, fisher_matrix=fisher_matrix)
