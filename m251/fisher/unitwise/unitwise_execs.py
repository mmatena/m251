"""TODO: Add title."""
import contextlib
import re

from absl import logging
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes
from del8.executables.models import checkpoints as ckpt_exec

from m251.models.bert import glue_classifier_execs as gc_exe

from . import unitwise


@executable.executable(
    default_bindings={
        "initializer": gc_exe.bert_initializer,
        "loader": ckpt_exec.checkpoint_loader,
        "builder": gc_exe.bert_builder,
    }
)
def fisher_computer(
    _initializer,
    _builder,
    _loader,
    num_examples,
    finetuned_ckpt_uuid=None,
    fisher_class_chunk_size=None,
    y_samples=None,
    fisher_device="gpu",
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

        computer = unitwise.UnitwiseFisherComputer(
            ft_model,
            total_examples=num_examples,
            fisher_device=fisher_device,
            y_samples=y_samples,
            class_chunk_size=fisher_class_chunk_size,
        )
        # NOTE: I don't really think the next binding will ever be used, but I'm
        # putting here out of paranoia.
        with scopes.binding_by_name_scope("model", computer):
            computer.compile()

    return computer


###############################################################################
