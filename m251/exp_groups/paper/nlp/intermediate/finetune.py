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

from m251.data.glue import glue
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

from m251.models import model_execs
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import glue_metric_execs as metrics_exe

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
        sequence_length,
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
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            #
            "epochs": keras_epochs,
            "steps_per_epoch": keras_steps_per_epoch,
        }


###############################################################################


HIGH_RESOURCE_TASKS = defs.HIGH_RESOURCE_TASKS
HIGH_RESOURCE_TRIALS = defs.HIGH_RESOURCE_TRIALS

LOW_RESOURCE_TASKS = defs.LOW_RESOURCE_TASKS
LOW_RESOURCE_TRIALS = defs.LOW_RESOURCE_TRIALS


def get_varying_param(task, trial_index):
    return {
        "task": task,
        "trial_index": trial_index,
        # Finetuning time given by min(10 epochs, 1M examples).
        "num_training_examples": min(1_000_000, 10 * NUM_GLUE_TRAIN_EXAMPLES[task]),
    }


@experiment.experiment(
    uuid="0d44993484ec4fa1bb089195ebce4dd8",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=(
        [
            get_varying_param(task, i)
            for task in HIGH_RESOURCE_TASKS
            for i in range(HIGH_RESOURCE_TRIALS)
        ]
        + [
            get_varying_param(task, i)
            for task in LOW_RESOURCE_TASKS
            for i in range(LOW_RESOURCE_TRIALS)
        ]
    ),
    fixed_params={
        "pretrained_model": "roberta-large",
        #
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 8,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_ckpt_saves": 10,
    },
    key_fields={
        "trial_index",
        "task",
    },
    bindings=[
        scopes.ArgNameBindingSpec("with_validation", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
    ],
)
class GlueFinetune(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="b10f85c98d4a4ef9adcd6b6f605e07a7",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=(
        [
            get_varying_param(task, i)
            for task in ["qnli", "mnli"]
            for i in range(HIGH_RESOURCE_TRIALS)
        ]
    ),
    fixed_params={
        "pretrained_model": "roberta-large",
        #
        "reg_type": "iso",
        "reg_strength": 1e-6,
        #
        "batch_size": 8,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_ckpt_saves": 10,
    },
    key_fields={
        "trial_index",
        "task",
    },
    bindings=[
        scopes.ArgNameBindingSpec("with_validation", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
    ],
)
class GlueFinetune_LowRegStrength(ExperimentAbc):
    pass
