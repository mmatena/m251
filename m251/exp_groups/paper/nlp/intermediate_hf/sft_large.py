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

from .fisher_large import FisherComputation_RobertLargeMnli_Rte_LastCkpt
from .fisher_large import FisherComputation_RobertLargeMnli_Rte_LastCkpt_AllVars
from ..intermediate.fisher import FisherComputation_LastCkpt


BAD_FINETUNE_RUN_UUIDS = frozenset(
    {
        "37dbf11090b047b2ba2e9996597e22ab",
        "ab6ce15a17ad4ea287c08093270ee494",
        "b8103c8e19054604a420b1ec2c1e4a15",
        "2b23839254934890acd9fab09803382c",
    }
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
            "glue_label_map_overrides": defs.LABEL_MAP_OVERRIDES,
            #
            "tfds_skip": self.tfds_skip,
            #
            "load_checkpoint_weights_by_name": True,
            "load_checkpoint_skip_mismatch": True,
            # "hf_back_compat": False,
            # "pretrained_body_only": True,
            #
            "checkpoint": self.checkpoint,
            #
            "hf_back_compat": True,
            "pretrained_body_only": True,
            "use_roberta_head": False,
        }


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
            if p.finetuned_run_uuid in BAD_FINETUNE_RUN_UUIDS:
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
    uuid="bb8a4f1b17914bb5b0ca555505dac84a",
    varying_params=functools.partial(
        create_lr_src_varying_params,
        fisher_exp=FisherComputation_LastCkpt,
        target_tasks=["rte"],
    ),
    fixed_params={
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 8,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_ckpt_saves": 10,
        #
        "pretrained_model": "roberta-large",
    },
    **COMMON_KWARGS,
)
class Large_Sft(ExperimentAbc):
    pass
