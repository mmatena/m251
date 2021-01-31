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
        num_task_epochs,
        num_ckpt_saves,
    ):
        pass

    def create_bindings(self):
        num_examples = self.num_task_epochs * NUM_GLUE_TRAIN_EXAMPLES[self.task]
        num_steps = int(round(num_examples / self.batch_size))

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


NUM_RTE_TRIALS = 5


@experiment.experiment(
    uuid="790ae156576b42ab95be420471cb2048",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=(
        [{"task": "mnli", "trial_index": 0}]
        + [{"task": "rte", "trial_index": i} for i in range(NUM_RTE_TRIALS)]
    ),
    fixed_params={
        "pretrained_model": "base",
        #
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 16,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_task_epochs": 2,
        "num_ckpt_saves": 4,
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
    uuid="4d6510859620483488ad6a05aebe2e36",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=[{"task": "rte", "trial_index": i} for i in range(NUM_RTE_TRIALS)],
    fixed_params={
        "pretrained_model": "base",
        #
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 16,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_task_epochs": 10,
        "num_ckpt_saves": 4,
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
class RteFinetune_10Epochs(ExperimentAbc):
    pass
