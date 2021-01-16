"""TODO: Add title."""
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.training import optimizers

from m251.data.glue import glue
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.execs import fisher_execs

from m251.models.bert import glue_classifier_execs as gc_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from . import defs

TASK_TO_CKPT = defs.TASK_TO_CKPT


@data_class.data_class()
class FisherParams(ParamsAbc):
    def __init__(
        self,
        #
        pretrained_model,
        task,
        #
        num_examples,
        #
        batch_size,
        sequence_length,
    ):
        pass

    def create_bindings(self):
        return {
            "compiled_fisher_computer": diag_execs.diagonal_fisher_computer,
            #
            "pretrained_model": self.pretrained_model,
            "tasks": [self.task],
            #
            "num_examples": self.num_examples,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }


###############################################################################


MAX_MNLI_RTE_FISHER_EXAMPLES = 4096


@experiment.experiment(
    uuid="0fcd7fa2ea794bff9ca3079361c091df",
    group=PaperExpGroup,
    params_cls=FisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "task": task,
            "pretrained_model": TASK_TO_CKPT[task],
            "num_examples": min(
                NUM_GLUE_TRAIN_EXAMPLES[task], MAX_MNLI_RTE_FISHER_EXAMPLES
            ),
        }
        for task in ["rte", "mnli"]
    ],
    fixed_params={
        "batch_size": 4,
        "sequence_length": 64,
    },
    key_fields={
        "pretrained_model",
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 3),
        #
        scopes.ArgNameBindingSpec("loader", gc_exe.bert_loader),
    ],
)
class FisherComputation_Base_MnliRte(ExperimentAbc):
    pass
