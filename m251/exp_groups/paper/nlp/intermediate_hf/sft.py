"""Sequential fine-tuning."""
import functools
import itertools

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

from .fisher import (
    FisherComputation_BertBase_LowResource_LastCkpt,
    FisherComputation_BertBaseFromMnliCkpt_LastCkpt,
    FisherComputation_BertBase_Squad2,
)


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
        checkpoint,
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
            #
            "load_checkpoint_weights_by_name": True,
            "load_checkpoint_skip_mismatch": True,
            "hf_back_compat": False,
            "pretrained_body_only": True,
            #
            "checkpoint": self.checkpoint,
        }


###############################################################################


def _get_params(exps_data, fisher_exp):
    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=fisher_exp.uuid)
    run_datas = [exps_data.get_run_data(run_id) for run_id in run_ids]
    checkpoints = [
        run_data.get_single_item_by_class(fisher_exp.params_cls)
        for run_data in run_datas
    ]
    return checkpoints


def create_lr_src_varying_params(
    exp,
    fisher_exp,
    target_tasks,
):
    with exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[fisher_exp.uuid])

    params = _get_params(exps_data, fisher_exp)

    varying_params = []
    for p in params:
        for task in target_tasks:
            if p.task == task:
                continue
            varying_params.append(
                {
                    "trial_index": p.trial_index,
                    "task": task,
                    "checkpoint": p.finetuned_ckpt_uuid,
                    "num_training_examples": min(
                        1_000_000, 10 * NUM_GLUE_TRAIN_EXAMPLES[task]
                    ),
                }
            )

    return varying_params


###############################################################################


COMMON_KWARGS = {
    "group": PaperExpGroup,
    "params_cls": FinetuneParams,
    "executable_cls": fitting.training_run,
    "key_fields": {
        "trial_index",
        "task",
        "pretrained_model",
        "checkpoint",
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
    ],
}


@experiment.experiment(
    uuid="5ace317bc7cc4e36a8925f1380bf3996",
    varying_params=functools.partial(
        create_lr_src_varying_params,
        fisher_exp=FisherComputation_BertBase_LowResource_LastCkpt,
        target_tasks=defs.LOW_RESOURCE_TASKS,
    ),
    fixed_params={
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 16,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_ckpt_saves": 10,
        #
        "pretrained_model": "bert-base-uncased",
    },
    **COMMON_KWARGS,
)
class GlueFinetune_BertBase_LrSrc_Sft(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="56df8e9b515c44d88bd548d51c48fddf",
    varying_params=[
        {
            "trial_index": trial_index,
            "task": lr_task,
            "pretrained_model": defs.TASK_TO_CKPT_BERT_BASE[hr_task],
            "num_training_examples": min(
                1_000_000, 10 * NUM_GLUE_TRAIN_EXAMPLES[lr_task]
            ),
        }
        for trial_index in range(defs.LOW_RESOURCE_TRIALS)
        for lr_task in defs.LOW_RESOURCE_TASKS
        for hr_task in defs.HIGH_RESOURCE_TASKS
    ],
    fixed_params={
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 16,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_ckpt_saves": 10,
        #
        "checkpoint": None,
    },
    **COMMON_KWARGS,
)
class GlueFinetune_BertBase_HrSrc_Sft(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="f87fae9c76954bd987916cf08c5d9ca2",
    varying_params=functools.partial(
        create_lr_src_varying_params,
        fisher_exp=FisherComputation_BertBaseFromMnliCkpt_LastCkpt,
        target_tasks=defs.LOW_RESOURCE_TASKS,
    ),
    fixed_params={
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 16,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_ckpt_saves": 10,
        #
        "pretrained_model": "bert-base-uncased",
    },
    **COMMON_KWARGS,
)
class GlueFinetune_BertBaseFromMnli_Sft(ExperimentAbc):
    pass


###############################################################################


SQUAD_KWARGS = {
    "group": PaperExpGroup,
    "params_cls": FinetuneParams,
    "executable_cls": fitting.training_run,
    "key_fields": {
        "trial_index",
        "task",
        "pretrained_model",
        "checkpoint",
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
        # #
        # scopes.ArgNameBindingSpec("pretrained_body_only", True),
        # scopes.ArgNameBindingSpec("pretrained_full_model", True),
        # scopes.ArgNameBindingSpec(
        #     "mergeable_model_pretrained_model", defs.BERT_BASE_MNLI_CKPT
        # ),
    ],
}


@experiment.experiment(
    uuid="b8eeac7b453d43cb89c177897fa4821b",
    varying_params=[
        {
            "trial_index": trial_index,
            "task": lr_task,
            "pretrained_model": defs.TASK_TO_CKPT_BERT_BASE["squad2"],
            "num_training_examples": min(
                1_000_000, 10 * NUM_GLUE_TRAIN_EXAMPLES[lr_task]
            ),
        }
        for trial_index in range(defs.LOW_RESOURCE_TRIALS)
        for lr_task in defs.LOW_RESOURCE_TASKS
    ],
    fixed_params={
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 16,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_ckpt_saves": 10,
        "checkpoint": None,
    },
    **COMMON_KWARGS,
)
class GlueFinetune_BertBase_Squad_Sft(ExperimentAbc):
    pass
