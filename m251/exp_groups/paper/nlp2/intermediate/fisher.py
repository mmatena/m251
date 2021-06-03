"""TODO: Add title."""
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.training import optimizers

from m251.data.glue import glue
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.execs import fisher_execs

from m251.models.bert import bert as bert_common
from m251.models.bert import glue_classifier_execs as gc_exe

from m251.models.bert import extractive_qa_execs as eqa_exe
from m251.data.qa import squad

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from . import defs
from .finetune import (
    GlueFinetune_BertBase,
)


MAX_EXAMPLES = 4096


@data_class.data_class()
class FromHfFisherParams(ParamsAbc):
    def __init__(
        self,
        #
        pretrained_model,
        task,
        #
        num_examples,
        #
        batch_size,
        sequence_length,
    ):
        pass

    def create_bindings(self):
        return {
            "compiled_fisher_computer": diag_execs.diagonal_fisher_computer,
            #
            "pretrained_model": self.pretrained_model,
            "tasks": [self.task],
            #
            "num_examples": self.num_examples,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            #
            "hf_back_compat": False,
            "glue_label_map_overrides": defs.LABEL_MAP_OVERRIDES,
        }

    @property
    def trial_index(self):
        return 0


@experiment.experiment(
    uuid="f42346afa2b5417db63e834b0b0a6c64",
    group=PaperExpGroup,
    params_cls=FromHfFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "task": task,
            "pretrained_model": defs.TASK_TO_CKPT_BERT_BASE[task],
            "num_examples": min(NUM_GLUE_TRAIN_EXAMPLES[task], MAX_EXAMPLES),
        }
        for task in defs.HIGH_RESOURCE_TASKS
    ],
    fixed_params={
        "batch_size": 4,
        "sequence_length": 64,
    },
    key_fields={
        "task",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 3),
        #
        scopes.ArgNameBindingSpec("loader", gc_exe.bert_loader),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
    ],
)
class FisherComputation_BertBase_HighResource(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="3ce7b1aa6c0a4f4f89e9047c96f068a9",
    group=PaperExpGroup,
    params_cls=FromHfFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "task": "squad2",
            "pretrained_model": defs.TASK_TO_CKPT_BERT_BASE["squad2"],
            "num_examples": MAX_EXAMPLES,
        }
    ],
    fixed_params={
        "batch_size": 1,
        "sequence_length": 512,
    },
    key_fields={
        "task",
        "pretrained_model",
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", 1),
        # eqa_exe.eqa_finetuning_model
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", squad.squad2_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("compiled_model", eqa_exe.eqa_finetuning_model),
        #
        scopes.ArgNameBindingSpec("tokenizer", bert_common.bert_tokenizer),
        scopes.ArgNameBindingSpec("initializer", eqa_exe.bert_initializer),
        scopes.ArgNameBindingSpec("loader", eqa_exe.bert_loader),
        scopes.ArgNameBindingSpec("builder", eqa_exe.bert_builder),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
    ],
)
class FisherComputation_BertBase_Squad2(ExperimentAbc):
    pass


###############################################################################
###############################################################################


@data_class.data_class()
class FisherParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        finetuned_run_uuid,
        finetuned_ckpt_uuid,
        #
        pretrained_model,
        task,
        #
        num_examples,
        #
        batch_size,
        sequence_length,
    ):
        pass

    def create_bindings(self):
        return {
            "compiled_fisher_computer": diag_execs.diagonal_fisher_computer,
            #
            "pretrained_model": self.pretrained_model,
            "tasks": [self.task],
            #
            "finetuned_run_uuid": self.finetuned_run_uuid,
            "finetuned_ckpt_uuid": self.finetuned_ckpt_uuid,
            #
            "num_examples": self.num_examples,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            #
            "hf_back_compat": False,
            "glue_label_map_overrides": defs.LABEL_MAP_OVERRIDES,
        }

    def create_preload_blob_uuids(self):
        return (self.finetuned_ckpt_uuid,)


@experiment.with_experiment_storages()
def create_varying_params_last_ckpt(
    exp,
    train_exp,
    max_examples,
):
    varying_params = []
    run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    for run_id in run_ids:
        run_params = train_exp.retrieve_run_params(run_id)

        checkpoints_summary = train_exp.retrieve_checkpoints_summary(run_id)

        ckpt_id = checkpoints_summary.checkpoint_uuids[-1]

        num_examples = min(NUM_GLUE_TRAIN_EXAMPLES[run_params.task], max_examples)

        varying_params.append(
            {
                "trial_index": run_params.trial_index,
                #
                "task": run_params.task,
                "pretrained_model": run_params.pretrained_model,
                #
                "finetuned_run_uuid": run_id,
                "finetuned_ckpt_uuid": ckpt_id,
                #
                "num_examples": num_examples,
            }
        )

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="4ba5f59b66cc4830975c7f681d06cc96",
    group=PaperExpGroup,
    params_cls=FisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params_last_ckpt,
        train_exp=GlueFinetune_BertBase,
        max_examples=MAX_EXAMPLES,
    ),
    fixed_params={
        "batch_size": 4,
        "sequence_length": 64,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 3),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
)
class FisherComputation_BertBase_LowResource_LastCkpt(ExperimentAbc):
    pass
