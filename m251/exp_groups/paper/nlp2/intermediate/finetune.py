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
        #
        tfds_skip=None,
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
            #
            "hf_back_compat": False,
            "glue_label_map_overrides": defs.LABEL_MAP_OVERRIDES,
            #
            "tfds_skip": self.tfds_skip,
        }


###############################################################################


def get_varying_param(task, trial_index, pretrained_model):
    num_examples = 10 * NUM_GLUE_TRAIN_EXAMPLES[task]
    assert num_examples < 1e6, f"Task {task} is not a low resource task."
    return {
        "pretrained_model": pretrained_model,
        "task": task,
        "trial_index": trial_index,
        # Finetuning time given by min(10 epochs, 1M examples).
        "num_training_examples": min(1_000_000, 10 * NUM_GLUE_TRAIN_EXAMPLES[task]),
    }


COMMON_KWARGS = {
    "group": PaperExpGroup,
    "params_cls": FinetuneParams,
    "executable_cls": fitting.training_run,
    "fixed_params": {
        "reg_type": "iso",
        "reg_strength": 0.0,
        #
        "batch_size": 16,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_ckpt_saves": 10,
    },
    "key_fields": {
        "trial_index",
        "task",
        "pretrained_model",
    },
    "bindings": [
        scopes.ArgNameBindingSpec("with_validation", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
}


###############################################################################


@experiment.experiment(
    uuid="02c9921f54ac4ff19d5008b74491989e",
    varying_params=[
        get_varying_param(task, i, "bert-base-uncased")
        for task in defs.LOW_RESOURCE_TASKS
        for i in range(defs.LOW_RESOURCE_TRIALS)
    ],
    **COMMON_KWARGS,
)
class GlueFinetune_BertBase(ExperimentAbc):
    pass
