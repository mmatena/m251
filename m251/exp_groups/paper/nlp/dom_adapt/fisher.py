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

from .finetune import Finetune_LowResource


TRAIN_EXAMPLES = target_tasks.TRAIN_EXAMPLES
TASK_TO_DAPT_NAME = target_tasks.TASK_TO_DAPT_NAME

LOW_RESOURCE_TASKS = ["chemprot", "acl_arc", "sci_erc"]


###############################################################################


@data_class.data_class()
class MlmTargetTaskFisherParams(ParamsAbc):
    def __init__(
        self,
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
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }


@experiment.experiment(
    uuid="da7b45b7dab243eba6e301f88ee45846",
    group=PaperExpGroup,
    params_cls=MlmTargetTaskFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "task": task,
            "num_examples": TRAIN_EXAMPLES[task],
            "pretrained_model": TASK_TO_DAPT_NAME[task],
        }
        for task in LOW_RESOURCE_TASKS
    ],
    fixed_params={
        "batch_size": 2,
        "sequence_length": 256,
        #
        "y_samples": 8,
    },
    key_fields={
        "pretrained_model",
        "task",
        "num_examples",
        "y_samples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.mlm_dataset),
        #
        scopes.ArgNameBindingSpec("compiled_model", mlm_execs.roberta_mlm_model),
        #
        scopes.ArgNameBindingSpec("tokenizer", bert_common.bert_tokenizer),
        scopes.ArgNameBindingSpec("initializer", mlm_execs.roberta_initializer),
        scopes.ArgNameBindingSpec("loader", mlm_execs.roberta_loader),
        scopes.ArgNameBindingSpec("builder", mlm_execs.roberta_builder),
    ],
)
class FisherComputation_MlmTargetTask(ExperimentAbc):
    pass


###############################################################################
###############################################################################


@data_class.data_class()
class MlmFisherParams(ParamsAbc):
    def __init__(
        self,
        #
        pretrained_model,
        task,
        num_examples,
        y_samples,
        #
        batch_size,
        sequence_length,
        #
        trial_index,
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
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }


S2ORC_TASKS = ["cs", "bio_med"]

S2ORC_TASK_TO_DAPT_NAME = {
    "bio_med": "allenai/biomed_roberta_base",
    "cs": "allenai/cs_roberta_base",
}


@experiment.experiment(
    uuid="6ce8d25bb2af469cba52551ea3659be3",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "trial_index": 0,
            "task": task,
            "pretrained_model": S2ORC_TASK_TO_DAPT_NAME[task],
            #
            "batch_size": 2,
            "y_samples": 8,
            "num_examples": 4096,
        }
        for task in S2ORC_TASKS
    ]
    + [
        {
            "trial_index": 0,
            "task": task,
            "pretrained_model": S2ORC_TASK_TO_DAPT_NAME[task],
            #
            "batch_size": 2,
            "y_samples": 1,
            "num_examples": 131072,
        }
        for task in S2ORC_TASKS
    ],
    fixed_params={
        "sequence_length": 256,
    },
    key_fields={
        "trial_index",
        #
        "pretrained_model",
        "task",
        #
        "num_examples",
        "y_samples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", s2orc.mlm_dataset),
        #
        scopes.ArgNameBindingSpec("compiled_model", mlm_execs.roberta_mlm_model),
        #
        scopes.ArgNameBindingSpec("tokenizer", bert_common.bert_tokenizer),
        scopes.ArgNameBindingSpec("initializer", mlm_execs.roberta_initializer),
        scopes.ArgNameBindingSpec("loader", mlm_execs.roberta_loader),
        scopes.ArgNameBindingSpec("builder", mlm_execs.roberta_builder),
    ],
)
class FisherComputation_MlmS2orc(ExperimentAbc):
    pass


###############################################################################
###############################################################################


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


###############################################################################


@experiment.experiment(
    uuid="592909a23ea041e3b3d99564df9aa076",
    group=PaperExpGroup,
    params_cls=TargetTaskFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        # NOTE: I accidentally computed the Fisher on the checkpoints trained from
        # the domain-adapted checkpoint. I don't think I need them anywhere. I'll
        # have filter them about by the "pretrained_model" attribute on the params.
        train_exp=Finetune_LowResource,
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
    ],
)
class FisherComputation_TargetTask(ExperimentAbc):
    pass
