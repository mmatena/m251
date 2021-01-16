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


TASKS = defs.TASKS
TASK_TO_NUM_TRIALS = defs.TASK_TO_NUM_TRIALS
TASK_TO_TRAIN_EXAMPLES = defs.TASK_TO_TRAIN_EXAMPLES


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
        num_training_examples,
        num_ckpt_saves,
    ):
        pass

    def create_bindings(self):
        num_steps = int(round(self.num_training_examples / self.batch_size))

        keras_epochs = self.num_ckpt_saves
        keras_steps_per_epoch = int(round(num_steps / keras_epochs))

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
            "epochs": keras_epochs,
            "steps_per_epoch": keras_steps_per_epoch,
        }


@experiment.experiment(
    uuid="e6e7181839a14ff9836670bc40c2212d",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {
            "trial_index": i,
            "task": task,
            "num_training_examples": 10 * TASK_TO_TRAIN_EXAMPLES[task],
        }
        for task in TASKS
        for i in range(TASK_TO_NUM_TRIALS[task])
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
        "num_ckpt_saves": 5,
    },
    key_fields={
        "trial_index",
        "task",
    },
    bindings=[
        scopes.ArgNameBindingSpec("with_validation", False),
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
class Finetune_10Epochs(ExperimentAbc):
    pass
