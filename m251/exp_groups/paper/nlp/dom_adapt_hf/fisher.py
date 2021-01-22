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
from m251.data.domains import s2orc

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.execs import fisher_execs

from m251.models import model_execs
from m251.models.bert import bert as bert_common
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import roberta_mlm_execs as mlm_execs

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .finetune import Finetune_Dapt_LowResource_FOR_REAL, Finetune_LowResource_FOR_REAL


TRAIN_EXAMPLES = target_tasks.TRAIN_EXAMPLES
TASK_TO_DAPT_NAME = target_tasks.TASK_TO_DAPT_NAME

LOW_RESOURCE_TASKS = ["chemprot", "acl_arc", "sci_erc"]


@data_class.data_class()
class TargetTaskFisherParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        checkpoint_index,
        #
        finetuned_run_uuid,
        finetuned_ckpt_uuid,
        #
        pretrained_model,
        task,
        num_examples,
        y_samples,
        #
        batch_size,
        sequence_length,
    ):
        pass

    def create_bindings(self):
        if self.num_examples is None:
            raise ValueError(
                "You need to set num_examples to full dataset size instead of setting it to None."
            )
        return {
            "compiled_fisher_computer": diag_execs.diagonal_fisher_computer,
            "y_samples": self.y_samples,
            #
            "pretrained_model": self.pretrained_model,
            "task": self.task,
            "tasks": [self.task],
            "num_examples": self.num_examples,
            #
            "finetuned_run_uuid": self.finetuned_run_uuid,
            "finetuned_ckpt_uuid": self.finetuned_ckpt_uuid,
            #
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }

    def create_preload_blob_uuids(self):
        return (self.finetuned_ckpt_uuid,)


@experiment.with_experiment_storages()
def create_varying_params(exp, train_exp, task_to_example_count, min_ckpt_index=0):
    varying_params = []
    run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    for run_id in run_ids:
        run_params = train_exp.retrieve_run_params(run_id)

        checkpoints_summary = train_exp.retrieve_checkpoints_summary(run_id)

        checkpoint_uuids = checkpoints_summary.checkpoint_uuids[min_ckpt_index:]
        for ckpt_index, ckpt_id in enumerate(checkpoint_uuids):

            varying_params.append(
                {
                    "trial_index": run_params.trial_index,
                    "checkpoint_index": min_ckpt_index + ckpt_index,
                    #
                    "task": run_params.task,
                    "pretrained_model": run_params.pretrained_model,
                    #
                    "finetuned_run_uuid": run_id,
                    "finetuned_ckpt_uuid": ckpt_id,
                    #
                    "num_examples": task_to_example_count[run_params.task],
                }
            )

    return varying_params


@experiment.experiment(
    uuid="a1b20682ce8c49348095043b099d5d3b",
    group=PaperExpGroup,
    params_cls=TargetTaskFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_Dapt_LowResource_FOR_REAL,
        task_to_example_count=TRAIN_EXAMPLES,
    ),
    fixed_params={
        "batch_size": 2,
        "sequence_length": 256,
        #
        "y_samples": None,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 4),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class FisherComputation_Dapt_TargetTask_FOR_REAL(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="91a38a642f5040a497cb88e6abecf395",
    group=PaperExpGroup,
    params_cls=TargetTaskFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_LowResource_FOR_REAL,
        task_to_example_count=TRAIN_EXAMPLES,
    ),
    fixed_params={
        "batch_size": 2,
        "sequence_length": 256,
        #
        "y_samples": None,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 4),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class FisherComputation_TargetTask_FOR_REAL(ExperimentAbc):
    pass


###############################################################################
###############################################################################


# Back compat reasons. The LHS was incorrectly named!
FisherComputation_TargetTask = FisherComputation_Dapt_TargetTask_FOR_REAL
