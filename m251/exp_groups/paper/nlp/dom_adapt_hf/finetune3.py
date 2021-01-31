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

from .pretrain import Pretrain_32768, Pretrain_More, Pretrain_32768_NoReg


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
        #
        checkpoint,
        pretrained_run_uuid,
    ):
        pass

    def create_bindings(self):
        num_examples = self.num_task_epochs * target_tasks.TRAIN_EXAMPLES[self.task]
        num_steps = int(round(num_examples / self.batch_size))

        keras_epochs = self.num_ckpt_saves
        keras_steps_per_epoch = int(round(num_steps / keras_epochs))

        print(self.pretrained_model)

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
            "checkpoint": self.checkpoint,
            #
            "epochs": keras_epochs,
            "steps_per_epoch": keras_steps_per_epoch,
        }


@experiment.with_experiment_storages()
def create_varying_params(exp, train_exp, tasks, num_trials):
    varying_params = []
    run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    for run_id in run_ids:
        run_params = train_exp.retrieve_run_params(run_id)

        checkpoints_summary = train_exp.retrieve_checkpoints_summary(run_id)

        checkpoint_uuid = checkpoints_summary.checkpoint_uuids[-1]
        for task in tasks:
            for trial_index in range(num_trials):
                varying_params.append(
                    {
                        "trial_index": trial_index,
                        #
                        "task": task,
                        "pretrained_model": run_params.pretrained_model,
                        #
                        "checkpoint": checkpoint_uuid,
                        "pretrained_run_uuid": run_id,
                    }
                )

    return varying_params


###############################################################################


# LOW_RESOURCE_TASKS = ["chemprot", "acl_arc", "sci_erc"]
LOW_RESOURCE_TASKS = ["acl_arc", "sci_erc"]

LOW_RESOURCE_NUM_TRIALS = 5
LOW_RESOURCE_EPOCHS = 10


@experiment.experiment(
    uuid="c0b63bee77c040a1ab1a5d162be48e04",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Pretrain_32768,
        tasks=LOW_RESOURCE_TASKS,
        num_trials=LOW_RESOURCE_NUM_TRIALS,
    ),
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
        #
        scopes.ArgNameBindingSpec("load_checkpoint_weights_by_name", True),
    ],
)
class Finetune_Pretrained32768_LowResource(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="a101d2cdee804fc0a438b1a903a3977d",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Pretrain_More,
        tasks=LOW_RESOURCE_TASKS,
        num_trials=LOW_RESOURCE_NUM_TRIALS,
    ),
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
        "checkpoint",
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
        #
        scopes.ArgNameBindingSpec("load_checkpoint_weights_by_name", True),
    ],
)
class Finetune_PretrainMore_LowResource(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="5f858f657f554e21b82cbebb7da3119c",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Pretrain_32768_NoReg,
        tasks=LOW_RESOURCE_TASKS,
        num_trials=LOW_RESOURCE_NUM_TRIALS,
    ),
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
        "checkpoint",
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
        #
        scopes.ArgNameBindingSpec("load_checkpoint_weights_by_name", True),
    ],
)
class Finetune_Pretrain32768NoReg(ExperimentAbc):
    pass
