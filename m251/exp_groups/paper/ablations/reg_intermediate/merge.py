"""TODO: Add title."""
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.evaluation import eval_execs
from del8.executables.training import optimizers

from m251.data.glue import glue
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.models.bert import glue_metric_execs as metrics_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import create_pairwise_weightings
from m251.exp_groups.paper.paper_group import ModelToMerge

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .finetune import GlueFinetune
from .fisher import FisherComputation


@data_class.data_class()
class MergeParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        models_to_merge,
        num_weightings,
        #
        sequence_length,
        batch_size,
        validation_examples,
        #
        pretrained_model,
    ):
        pass

    def create_bindings(self):
        return {
            "mergeable_model": diag_execs.diagonal_mergeable_model_from_checkpoint,
            "model_merger": diag_execs.diagonal_model_merger,
            #
            "checkpoint_to_fisher_matrix_uuid": self.get_checkpoint_to_fisher_matrix_uuid(),
            "weightings": create_pairwise_weightings(self.num_weightings),
            #
            "checkpoints": [m.model_checkpoint_uuid for m in self.models_to_merge],
            "checkpoint_tasks": [m.task for m in self.models_to_merge],
            #
            "task": self.models_to_merge[0].task,
            "tasks": [m.task for m in self.models_to_merge],
            #
            "pretrained_model": self.pretrained_model,
            #
            "num_examples": self.validation_examples,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }

    def get_checkpoint_to_fisher_matrix_uuid(self):
        return {
            m.model_checkpoint_uuid: m.fisher_matrix_uuid for m in self.models_to_merge
        }

    def create_preload_blob_uuids(self):
        dikt = self.get_checkpoint_to_fisher_matrix_uuid()
        return tuple(set(dikt.keys()) | set(dikt.values()))


###############################################################################


def _filter_run_params_by_task(run_params, task):
    return {
        run_uuid: run_param
        for run_uuid, run_param in run_params.items()
        if run_param.task == task
    }


def _run_params_to_mtm(run_uuid, run_params, fishers):
    return ModelToMerge(
        task=run_params.task,
        train_run_uuid=run_params.finetuned_run_uuid,
        fisher_run_uuid=run_uuid,
        model_checkpoint_uuid=run_params.finetuned_ckpt_uuid,
        fisher_matrix_uuid=fishers[run_uuid],
    )


def _map_run_uuid_to_fisher_matrix_uuid(fisher_exp, run_uuids):
    return {
        run_uuid: fisher_exp.retrieve_single_item_by_class(
            fisher_execs.SavedFisherMatrix, run_uuid
        ).blob_uuid
        for run_uuid in run_uuids
    }


@experiment.with_experiment_storages()
def create_varying_params(
    exp,
    fisher_exp,
    target_task="rte",
    donor_task="mnli",
):
    run_uuids = fisher_exp.retrieve_run_uuids(RunState.FINISHED)
    run_params = {
        run_uuid: fisher_exp.retrieve_run_params(run_uuid) for run_uuid in run_uuids
    }

    fishers = _map_run_uuid_to_fisher_matrix_uuid(fisher_exp, run_uuids)

    target_run_params = _filter_run_params_by_task(run_params, target_task)
    donor_run_params = _filter_run_params_by_task(run_params, donor_task)

    varying_params = []

    for target_run_uuid, target_param in target_run_params.items():
        target_mtm = _run_params_to_mtm(target_run_uuid, target_param, fishers)
        for donor_run_uuid, donor_param in donor_run_params.items():
            donor_mtm = _run_params_to_mtm(donor_run_uuid, donor_param, fishers)
            varying_params.append(
                {
                    "trial_index": target_param.trial_index,
                    "models_to_merge": [target_mtm, donor_mtm],
                }
            )

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="3d1e69adb7ed40b78872832c78d99c07",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        fisher_exp=FisherComputation,
    ),
    fixed_params={
        "num_weightings": 51,
        #
        "validation_examples": 2048,
        "sequence_length": 64,
        "batch_size": 2048,
        #
        "pretrained_model": "base",
    },
    key_fields={
        "trial_index",
        "models_to_merge",
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
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
    ],
)
class Merge(ExperimentAbc):
    pass
