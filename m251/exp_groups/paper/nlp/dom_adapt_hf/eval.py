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

from m251.data.domains import target_tasks

from m251.models import model_execs
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import glue_metric_execs as metrics_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .finetune import Finetune_Dapt_LowResource, Finetune_Dapt_LowResource_FOR_REAL


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
    uuid="54a3a10c2dbd4c08ac46cb983a4b4882",
    group=PaperExpGroup,
    params_cls=EvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    varying_params=functools.partial(
        create_varying_eval_params,
        train_exp=Finetune_Dapt_LowResource,
    ),
    fixed_params={
        "sequence_length": 256,
        "num_examples": 2048,
        "batch_size": 128,
    },
    key_fields={
        "trial_index",
        "task",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", target_tasks.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class Eval_LowResource_Dapt_All(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="e5c5d77393934d619866cc66c7a92bcf",
    group=PaperExpGroup,
    params_cls=EvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    varying_params=functools.partial(
        create_varying_eval_params,
        train_exp=Finetune_Dapt_LowResource_FOR_REAL,
    ),
    fixed_params={
        "sequence_length": 256,
        "num_examples": 2048,
        "batch_size": 128,
    },
    key_fields={
        "trial_index",
        "task",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", target_tasks.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class Finetune_Dapt_LowResource_All_FOR_REAL(ExperimentAbc):
    pass
