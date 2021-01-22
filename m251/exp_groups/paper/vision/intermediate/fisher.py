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

from m251.data.image import image_classification
from m251.models import model_execs
from m251.models.simclr import simclr
from m251.models.simclr import simclr_classifier_execs as sc_exe

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.execs import fisher_execs

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from . import defs


@data_class.data_class()
class FisherParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        pretrained_model,
        task,
        #
        num_examples,
        #
        batch_size,
        image_size,
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
            "image_size": self.image_size,
            "batch_size": self.batch_size,
        }


###############################################################################


MAX_FISHER_EXAMPLES = 8192


@experiment.experiment(
    uuid="a18703d454264336b0863727d9de38b7",
    group=PaperExpGroup,
    params_cls=FisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "trial_index": trial_index,
            "task": task,
            "pretrained_model": defs.TASK_TO_FINETUNED_MODEL[task],
            "num_examples": min(defs.TASK_TO_TRAIN_EXAMPLES[task], MAX_FISHER_EXAMPLES),
        }
        for task in defs.TASKS
        for trial_index in range(5)
    ],
    fixed_params={
        "batch_size": 8,
        "image_size": simclr.IMAGE_SIZE,
    },
    key_fields={
        "trial_index",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", 1),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("initializer", sc_exe.simclr_initializer),
        scopes.ArgNameBindingSpec("loader", sc_exe.simclr_loader),
        scopes.ArgNameBindingSpec("builder", sc_exe.simclr_builder),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
        #
        scopes.ArgNameBindingSpec("color_distort", False),
        scopes.ArgNameBindingSpec("test_crop", False),
    ],
)
class FisherComputation_1x(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="4b585375b7474b33905ad9fbdaeea75b",
    group=PaperExpGroup,
    params_cls=FisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "trial_index": 0,
            "task": task,
            "pretrained_model": defs.TASK_TO_FINETUNED_MODEL[task],
            "num_examples": min(defs.TASK_TO_TRAIN_EXAMPLES[task], 4096),
        }
        for task in defs.TASKS
    ],
    fixed_params={
        "batch_size": 2,
        "image_size": simclr.IMAGE_SIZE,
    },
    key_fields={
        "trial_index",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 4),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("initializer", sc_exe.simclr_initializer),
        scopes.ArgNameBindingSpec("loader", sc_exe.simclr_loader),
        scopes.ArgNameBindingSpec("builder", sc_exe.simclr_builder),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
        #
        scopes.ArgNameBindingSpec("color_distort", False),
        scopes.ArgNameBindingSpec("test_crop", True),
    ],
)
class FisherComputation_1x_Exact(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="542ae25fe9a143f3a3c4bb7dc03e06d9",
    group=PaperExpGroup,
    params_cls=FisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "trial_index": 0,
            "task": task,
            "pretrained_model": defs.TASK_TO_4X_FINETUNED_MODEL[task],
            "num_examples": min(defs.TASK_TO_TRAIN_EXAMPLES[task], 4096),
        }
        for task in reversed(defs.TASKS)
    ],
    fixed_params={
        "batch_size": 1,
        "image_size": simclr.IMAGE_SIZE,
    },
    key_fields={
        "trial_index",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", 8),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("initializer", sc_exe.simclr_initializer),
        scopes.ArgNameBindingSpec("loader", sc_exe.simclr_loader),
        scopes.ArgNameBindingSpec("builder", sc_exe.simclr_builder),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
        #
        scopes.ArgNameBindingSpec("color_distort", False),
        scopes.ArgNameBindingSpec("test_crop", True),
    ],
)
class FisherComputation_4x_Sampling(ExperimentAbc):
    pass
