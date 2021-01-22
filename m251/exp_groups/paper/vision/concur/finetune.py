"""TODO: Add title."""
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.evaluation import eval_execs
from del8.executables.models import checkpoints as ckpt_exec
from del8.executables.training import fitting
from del8.executables.training import optimizers

from m251.data.image import image_classification
from m251.models import model_execs
from m251.models.simclr import simclr
from m251.models.simclr import simclr_classifier_execs as sc_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from . import defs


@data_class.data_class()
class FinetuneParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        pretrained_model,
        task,
        #
        reg_type,
        reg_strength,
        #
        learning_rate,
        image_size,
        batch_size,
        #
        steps_per_epoch,
        epochs,
        #
        train_examples,
        train_skip_examples,
        #
        validation_examples,
    ):
        pass

    def create_bindings(self):

        return {
            "pretrained_model": self.pretrained_model,
            "tasks": [self.task],
            #
            "regularizer": model_execs.regularize_body_l2_from_initial,
            "reg_strength": self.reg_strength,
            #
            "learning_rate": self.learning_rate,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
            #
            "epochs": self.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            #
            "train_num_examples": self.train_examples,
            "train_skip_examples": self.train_skip_examples,
            #
            "validation_num_examples": self.validation_examples,
        }


def _percent(task, percent):
    return int(percent * defs.TASK_TO_TRAIN_EXAMPLES[task] / 100)


@experiment.experiment(
    uuid="021024afacd046d788086b70112ab4b4",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {
            "trial_index": task_index,
            "task": task,
            "train_examples": _percent(task, percent),
            "train_skip_examples": subset * _percent(task, percent),
        }
        for task in ["cifar100"]
        for task_index in range(1)
        for subset in range(2)
        for percent in [1]
    ],
    fixed_params={
        "pretrained_model": "r50_1x",
        #
        "reg_type": "iso",
        "reg_strength": 1e-6,
        #
        "image_size": simclr.IMAGE_SIZE,
        #
        "batch_size": 32,
        "learning_rate": 4e-5,
        #
        "steps_per_epoch": 50,
        "epochs": 10,
        #
        "validation_examples": 2048,
    },
    key_fields={
        "trial_index",
        "task",
        "train_examples",
        "train_skip_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("with_validation", True),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("compiled_model", sc_exe.simclr_finetuning_model),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
    ],
)
class Finetune_Subsets(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="021024afacd046d788086b70112ab4b4",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {
            "trial_index": task_index,
            "task": task,
            "train_examples": _percent(task, percent),
            "train_skip_examples": subset * _percent(task, percent),
        }
        for task in ["cifar100"]
        for task_index in range(1)
        for subset in range(2)
        for percent in [1]
    ],
    fixed_params={
        "pretrained_model": "r50_1x",
        #
        "reg_type": "iso",
        "reg_strength": 1e-6,
        #
        "image_size": simclr.IMAGE_SIZE,
        #
        "batch_size": 32,
        "learning_rate": 4e-5,
        #
        "steps_per_epoch": 50,
        "epochs": 10,
        #
        "validation_examples": 2048,
    },
    key_fields={
        "trial_index",
        "task",
        "train_examples",
        "train_skip_examples",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("with_validation", True),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("compiled_model", sc_exe.simclr_finetuning_model),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
    ],
)
class Finetune_Subsets2(ExperimentAbc):
    pass
