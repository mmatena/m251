"""TODO: Add title."""
import functools

from absl import logging

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs

from m251.data.text import wiki40b
from m251.fisher.diagonal import diagonal_execs
from m251.fisher.execs import fisher_execs
from m251.models.bert import bert_mlm_execs as mlm_exe

from ..group import BertMergingPrelimsGroup


@data_class.data_class()
class MlmFisherParams(object):
    def __init__(
        self,
        dataset,
        fisher_type,
        diagonal_y_samples,
        pretrained_model,
        num_examples,
        sequence_length,
        batch_size,
    ):
        pass

    def create_binding_specs(self):
        if self.fisher_type == "diagonal":
            fisher_bindings = [
                scopes.ArgNameBindingSpec(
                    "compiled_fisher_computer", diagonal_execs.diagonal_fisher_computer
                ),
                scopes.ArgNameBindingSpec("fisher_type", self.fisher_type),
                scopes.ArgNameBindingSpec("y_samples", self.diagonal_y_samples),
            ]
        else:
            raise ValueError(f"Invalid fisher_type {self.fisher_type}.")

        if self.dataset == "wiki40b":
            data_set_bindings = [
                scopes.ArgNameBindingSpec("dataset", wiki40b.wiki40b_mlm_dataset),
            ]
        else:
            raise ValueError(f"Invalid dataset {self.dataset}.")

        return [
            scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
            #
            scopes.ArgNameBindingSpec("num_examples", self.num_examples),
            scopes.ArgNameBindingSpec("sequence_length", self.sequence_length),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
            *fisher_bindings,
            *data_set_bindings,
        ]


# Base, last checkpoint.
@experiment.experiment(
    uuid="5216d0a048764698873feebd10d3495a",
    group=BertMergingPrelimsGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {"pretrained_model": "base"},
        {"pretrained_model": "large"},
    ],
    fixed_params={
        "dataset": "wiki40b",
        "num_examples": 32768,
        #
        "fisher_type": "diagonal",
        "diagonal_y_samples": 8,
        #
        "sequence_length": 256,
        "batch_size": 1,
    },
    key_fields={
        "pretrained_model",
        #
        "dataset",
        "num_examples",
        #
        "fisher_type",
        "diagonal_y_samples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("initializer", mlm_exe.bert_initializer),
        scopes.ArgNameBindingSpec("loader", mlm_exe.bert_loader),
        scopes.ArgNameBindingSpec("builder", mlm_exe.bert_builder),
        scopes.ArgNameBindingSpec("metrics", mlm_exe.bert_mlm_metrics),
        #
        scopes.ArgNameBindingSpec("split", "train"),
        scopes.ArgNameBindingSpec("shuffle", True),
        scopes.ArgNameBindingSpec("repeat", False),
    ],
)
class MlmFisher_Bert(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )
