"""TODO: Add title."""
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
    y_samples=None,
):
    with scopes.binding_by_name_scope("checkpoint", finetuned_ckpt_uuid):
        ft_model = _initializer()
        with scopes.binding_by_name_scope("model", ft_model):
            ft_model = _builder(ft_model)
            ft_model = _loader(ft_model)

        computer = diagonal.DiagonalFisherComputer(
            ft_model, total_examples=num_examples, y_samples=y_samples
        )
        # NOTE: I don't really think the next binding will ever be used, but I'm
        # putting here out of paranoia.
        with scopes.binding_by_name_scope("model", computer):
            computer.compile()

    return computer
