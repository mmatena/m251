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
from del8.executables.training import optimizers

from m251.data.glue import glue
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.diagonal import variational_diagonal_execs as vardiag_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.models.bert import glue_metric_execs as metrics_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import create_pairwise_weightings
from m251.exp_groups.paper.paper_group import ModelToMerge

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .finetune import GlueFinetune, RteFinetune_10Epochs
from .fisher import FisherComputation, FisherComputation_Rte10Epochs


@data_class.data_class()
class MergeParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        target_ckpt_index,
        donor_ckpt_index,
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
    return [run_param for run_param in run_params if run_param.task == task]


def _run_params_to_mtm(run_params, fishers):
    return ModelToMerge(
        task=run_params.task,
        train_run_uuid=run_params.finetuned_run_uuid,
        fisher_run_uuid=run_params.run_uuid,
        model_checkpoint_uuid=run_params.finetuned_ckpt_uuid,
        fisher_matrix_uuid=fishers[run_params.run_uuid],
    )


def _map_run_uuid_to_fisher_matrix_uuid(run_uuids, run_datas):
    return {
        run_uuid: run_data.get_single_item_by_class(
            fisher_execs.SavedFisherMatrix
        ).blob_uuid
        for run_uuid, run_data in zip(run_uuids, run_datas)
    }


def _get_ckpt_index(mtm, ckpt_summaries):
    ckpt_summary = ckpt_summaries[mtm.train_run_uuid]
    ckpt_index = ckpt_summary.checkpoint_uuids.index(mtm.model_checkpoint_uuid)
    return ckpt_index


def create_varying_params(
    exp,
    train_exp,
    fisher_exp,
    target_task="rte",
    donor_task="mnli",
):
    with fisher_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[fisher_exp.uuid, train_exp.uuid]
        )

    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=fisher_exp.uuid)
    run_datas = [exps_data.get_run_data(run_id) for run_id in run_ids]

    run_params = [
        run_data.get_single_item_by_class(fisher_exp.params_cls)
        for run_data in run_datas
    ]
    target_run_params = _filter_run_params_by_task(run_params, target_task)
    donor_run_params = _filter_run_params_by_task(run_params, donor_task)

    fishers = _map_run_uuid_to_fisher_matrix_uuid(run_ids, run_datas)

    train_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=train_exp.uuid)
    ckpt_summaries = {
        run_id: exps_data.get_run_data(run_id).get_single_item_by_class(
            checkpoints.CheckpointsSummary
        )
        for run_id in train_run_ids
    }

    varying_params = []

    for target_param in target_run_params:
        target_mtm = _run_params_to_mtm(target_param, fishers)
        for donor_param in donor_run_params:
            donor_mtm = _run_params_to_mtm(donor_param, fishers)
            varying_params.append(
                {
                    "trial_index": target_param.trial_index,
                    "target_ckpt_index": _get_ckpt_index(target_mtm, ckpt_summaries),
                    "donor_ckpt_index": _get_ckpt_index(donor_mtm, ckpt_summaries),
                    "models_to_merge": [target_mtm, donor_mtm],
                }
            )

    return varying_params


def _get_infos_for_task(exps_data, train_exp, fisher_exp, task):
    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=fisher_exp.uuid)
    run_datas = [exps_data.get_run_data(run_id) for run_id in run_ids]

    run_params = [
        run_data.get_single_item_by_class(fisher_exp.params_cls)
        for run_data in run_datas
    ]
    run_params = _filter_run_params_by_task(run_params, task)

    fishers = _map_run_uuid_to_fisher_matrix_uuid(run_ids, run_datas)

    train_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=train_exp.uuid)
    ckpt_summaries = {
        run_id: exps_data.get_run_data(run_id).get_single_item_by_class(
            checkpoints.CheckpointsSummary
        )
        for run_id in train_run_ids
    }
    return run_params, fishers, ckpt_summaries


def create_varying_params_sep_exps(
    exp,
    target_train_exp,
    target_fisher_exp,
    donor_train_exp,
    donor_fisher_exp,
    target_task="rte",
    donor_task="mnli",
):
    with target_train_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[
                target_train_exp.uuid,
                target_fisher_exp.uuid,
                donor_train_exp.uuid,
                donor_fisher_exp.uuid,
            ]
        )

    target_run_params, target_fishers, target_ckpt_summaries = _get_infos_for_task(
        exps_data, target_train_exp, target_fisher_exp, target_task
    )
    donor_run_params, donor_fishers, donor_ckpt_summaries = _get_infos_for_task(
        exps_data, donor_train_exp, donor_fisher_exp, donor_task
    )

    varying_params = []

    for target_param in target_run_params:
        target_mtm = _run_params_to_mtm(target_param, target_fishers)
        for donor_param in donor_run_params:
            donor_mtm = _run_params_to_mtm(donor_param, donor_fishers)
            varying_params.append(
                {
                    "trial_index": target_param.trial_index,
                    "target_ckpt_index": _get_ckpt_index(
                        target_mtm, target_ckpt_summaries
                    ),
                    "donor_ckpt_index": _get_ckpt_index(
                        donor_mtm, donor_ckpt_summaries
                    ),
                    "models_to_merge": [target_mtm, donor_mtm],
                }
            )

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="ee6ab84eb3404bde9faeca40df5264c4",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=GlueFinetune,
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
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        #
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
    ],
)
class Merge(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="6baac9b53e7d4c7e8250b3a753ba14fd",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params_sep_exps,
        target_train_exp=RteFinetune_10Epochs,
        target_fisher_exp=FisherComputation_Rte10Epochs,
        donor_train_exp=GlueFinetune,
        donor_fisher_exp=FisherComputation,
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
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        #
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
    ],
)
class Merge_Rte10Epochs(ExperimentAbc):
    pass
