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

from m251.data.domains import target_tasks

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
        num_examples = self.num_task_epochs * target_tasks.TRAIN_EXAMPLES[self.task]
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

TASK_TO_DAPT_NAME = target_tasks.TASK_TO_DAPT_NAME

LOW_RESOURCE_TASKS = ["chemprot", "acl_arc", "sci_erc"]

LOW_RESOURCE_NUM_TRIALS = 5
LOW_RESOURCE_EPOCHS = 10


@experiment.experiment(
    uuid="78a6437ffd8f41ddb02d13463adf56df",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {
            "trial_index": i,
            "task": task,
            "pretrained_model": TASK_TO_DAPT_NAME[task],
        }
        for i in range(LOW_RESOURCE_NUM_TRIALS)
        for task in LOW_RESOURCE_TASKS
    ],
    fixed_params={
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 8,
        "learning_rate": 1e-5,
        "sequence_length": 256,
        #
        "num_task_epochs": LOW_RESOURCE_EPOCHS,
        "num_ckpt_saves": LOW_RESOURCE_EPOCHS,
    },
    key_fields={
        "trial_index",
        "task",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("with_validation", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class Finetune_Dapt_LowResource_FOR_REAL(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="fa20c708ff184813be2d56fc00c43710",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {
            "trial_index": i,
            "task": task,
            "pretrained_model": "roberta-base",
        }
        for i in range(LOW_RESOURCE_NUM_TRIALS)
        for task in LOW_RESOURCE_TASKS
    ],
    fixed_params={
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 8,
        "learning_rate": 1e-5,
        "sequence_length": 256,
        #
        "num_task_epochs": LOW_RESOURCE_EPOCHS,
        "num_ckpt_saves": LOW_RESOURCE_EPOCHS,
    },
    key_fields={
        "trial_index",
        "task",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("with_validation", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class Finetune_LowResource_FOR_REAL(ExperimentAbc):
    pass


###############################################################################
###############################################################################


# Back compat reasons. The LHS was incorrectly named!
Finetune_LowResource = Finetune_Dapt_LowResource_FOR_REAL

# Back compat reasons. The LHS was incorrectly named!
Finetune_Dapt_LowResource = Finetune_LowResource_FOR_REAL
