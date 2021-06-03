"""TODO: Add title."""
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.evaluation import eval_execs
from del8.executables.models import checkpoints
from del8.executables.training import fitting
from del8.executables.training import optimizers

from m251.data.glue import glue

from m251.models import model_execs
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import glue_metric_execs as metrics_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .sft import (
    GlueFinetune_BertBase_LrSrc_Sft,
    GlueFinetune_BertBase_HrSrc_Sft,
    GlueFinetune_BertBaseFromMnli_Sft,
    GlueFinetune_BertBase_Squad_Sft,
)


@data_class.data_class()
class EvalParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        checkpoints_summary,
        #
        pretrained_model,
        task,
        #
        num_examples,
        batch_size,
        sequence_length,
    ):
        pass

    def create_bindings(self):
        print(self.pretrained_model)
        return {
            "checkpoints_summary": self.checkpoints_summary,
            "pretrained_model": self.pretrained_model,
            "tasks": [self.task],
            "num_examples": self.num_examples,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
        }

    def create_preload_blob_uuids(self):
        return self.checkpoints_summary.checkpoint_uuids


# checkpoints.CheckpointsSummary


def create_varying_eval_params(_, train_exp):
    train_exp.no_gcs_connect = True
    with train_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[train_exp.uuid])
    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=train_exp.uuid)

    varying_params = []
    for run_id in run_ids:
        run_data = exps_data.get_run_data(run_id)
        run_params = run_data.get_single_item_by_class(train_exp.params_cls)
        ckpt_summary = run_data.get_single_item_by_class(checkpoints.CheckpointsSummary)
        ckpt_summary = checkpoints.CheckpointsSummary(
            checkpoint_uuids=ckpt_summary.checkpoint_uuids[-1:],
            run_extra_identifier=ckpt_summary.run_extra_identifier,
        )
        varying_params.append(
            {
                "trial_index": run_params.trial_index,
                "checkpoints_summary": ckpt_summary,
                "pretrained_model": run_params.pretrained_model,
                "task": run_params.task,
            }
        )

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="8352815cfb9b49248ce92c0968b20d65",
    group=PaperExpGroup,
    params_cls=EvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    varying_params=functools.partial(
        create_varying_eval_params,
        train_exp=GlueFinetune_BertBase_LrSrc_Sft,
    ),
    fixed_params={
        "sequence_length": 64,
        "num_examples": 2048,
        "batch_size": 512,
    },
    key_fields={
        "trial_index",
        "task",
        "checkpoints_summary",
    },
    bindings=[
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", glue.glue_robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
)
class Eval_LrSrc_Sft_LastCkpt(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="5a40ad97369246c790f517d183b17278",
    group=PaperExpGroup,
    params_cls=EvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    varying_params=functools.partial(
        create_varying_eval_params,
        train_exp=GlueFinetune_BertBase_HrSrc_Sft,
    ),
    fixed_params={
        "sequence_length": 64,
        "num_examples": 2048,
        "batch_size": 512,
    },
    key_fields={
        "trial_index",
        "task",
        "checkpoints_summary",
    },
    bindings=[
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", glue.glue_robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
)
class Eval_HrSrc_Sft_LastCkpt(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="5e44d75234cf4a8ab8d223a04597680f",
    group=PaperExpGroup,
    params_cls=EvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    varying_params=functools.partial(
        create_varying_eval_params,
        train_exp=GlueFinetune_BertBaseFromMnli_Sft,
    ),
    fixed_params={
        "sequence_length": 64,
        "num_examples": 2048,
        "batch_size": 512,
    },
    key_fields={
        "trial_index",
        "task",
        "checkpoints_summary",
    },
    bindings=[
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", glue.glue_robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
)
class EvalBertBaseFromMnli_Sft_LastCkpt(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="d7f6aee1d404412d8ace0784a1e0cd25",
    group=PaperExpGroup,
    params_cls=EvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    varying_params=functools.partial(
        create_varying_eval_params,
        train_exp=GlueFinetune_BertBase_Squad_Sft,
    ),
    fixed_params={
        "sequence_length": 64,
        "num_examples": 2048,
        "batch_size": 512,
    },
    key_fields={
        "trial_index",
        "task",
        "checkpoints_summary",
    },
    bindings=[
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", glue.glue_robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
)
class EvalGlueFinetune_BertBase_Squad_Sft(ExperimentAbc):
    pass
