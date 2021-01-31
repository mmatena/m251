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

from m251.data.domains import target_tasks
from m251.data.glue import glue

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.models.bert import glue_metric_execs as metrics_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import create_pairwise_weightings
from m251.exp_groups.paper.paper_group import ModelToMerge

from m251.exp_groups.paper.paper_group import PaperExpGroup

from m251.exp_groups.paper.results import utils as result_utils

from ..dom_adapt.fisher import FisherComputation_MlmS2orc

from .finetune2 import Finetune_ROBERTA_LowResource
from .fisher2 import FisherComputation_ROBERTA_TargetTasks


TASK_TO_DAPT_NAME = target_tasks.TASK_TO_DAPT_NAME

get_single_score = result_utils.get_single_score


@data_class.data_class()
class MergeParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        target_ckpt_index,
        #
        models_to_merge,
        num_weightings,
        #
        sequence_length,
        batch_size,
        validation_examples,
        #
        pretrained_mlm_model,
        normalize_fishers=False,
    ):
        pass

    def create_bindings(self):
        return {
            "mergeable_model": diag_execs.diagonal_mergeable_model_from_checkpoint_or_pretrained,
            "model_merger": diag_execs.diagonal_model_merger,
            #
            "checkpoint_to_fisher_matrix_uuid": self.get_checkpoint_to_fisher_matrix_uuid(),
            "weightings": create_pairwise_weightings(self.num_weightings),
            #
            "normalize_fishers": self.normalize_fishers,
            #
            "checkpoints": [m.model_checkpoint_uuid for m in self.models_to_merge],
            "checkpoint_tasks": [m.task for m in self.models_to_merge],
            #
            "task": self.models_to_merge[0].task,
            "tasks": [m.task for m in self.models_to_merge],
            #
            "pretrained_model": self.pretrained_mlm_model,
            #
            "num_examples": self.validation_examples,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }

    def get_checkpoint_to_fisher_matrix_uuid(self):
        key = (
            lambda m: m.model_checkpoint_uuid
            if m.model_checkpoint_uuid
            else self.pretrained_mlm_model
        )
        return {key(m): m.fisher_matrix_uuid for m in self.models_to_merge}

    def create_preload_blob_uuids(self):
        dikt = self.get_checkpoint_to_fisher_matrix_uuid()
        return tuple(
            (set(dikt.keys()) | set(dikt.values())) - {self.pretrained_mlm_model}
        )


###############################################################################


def _finetuned_to_mtm(run_params, fishers):
    return ModelToMerge(
        task=run_params.task,
        train_run_uuid=run_params.finetuned_run_uuid,
        fisher_run_uuid=run_params.run_uuid,
        model_checkpoint_uuid=run_params.finetuned_ckpt_uuid,
        fisher_matrix_uuid=fishers[run_params.run_uuid],
    )


def _pretrained_to_mtm(run_params, fishers):
    return ModelToMerge(
        task=run_params.task,
        train_run_uuid=None,
        fisher_run_uuid=run_params.run_uuid,
        model_checkpoint_uuid=None,
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


def _get_infos_for_task(exps_data, fisher_exp):
    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=fisher_exp.uuid)
    run_datas = [exps_data.get_run_data(run_id) for run_id in run_ids]

    run_params = [
        run_data.get_single_item_by_class(fisher_exp.params_cls)
        for run_data in run_datas
    ]

    fishers = _map_run_uuid_to_fisher_matrix_uuid(run_ids, run_datas)

    return run_params, fishers


def _get_ckpt_summaries(exps_data, train_exp):
    train_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=train_exp.uuid)
    ckpt_summaries = {
        run_id: exps_data.get_run_data(run_id).get_single_item_by_class(
            checkpoints.CheckpointsSummary
        )
        for run_id in train_run_ids
    }
    return ckpt_summaries


def create_varying_params(
    exp,
    train_exp,
    target_fisher_exp,
    donor_fisher_exp,
    mlm_fisher_examples,
    ckpt_index=None,
    task_to_dapt_name=TASK_TO_DAPT_NAME,
):
    with target_fisher_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[
                train_exp.uuid,
                target_fisher_exp.uuid,
                donor_fisher_exp.uuid,
            ]
        )

    target_run_params, target_fishers = _get_infos_for_task(
        exps_data, target_fisher_exp
    )
    target_ckpt_summaries = _get_ckpt_summaries(exps_data, train_exp)

    donor_run_params, donor_fishers = _get_infos_for_task(exps_data, donor_fisher_exp)
    # dapt_to_donor_run_params = {p.pretrained_model: p for p in donor_run_params}
    dapt_to_donor_run_params = {
        p.pretrained_model: p
        for p in donor_run_params
        if p.num_examples == mlm_fisher_examples
    }

    varying_params = []

    for target_param in target_run_params:
        target_mtm = _finetuned_to_mtm(target_param, target_fishers)
        dapt = task_to_dapt_name[target_mtm.task]
        donor_params = dapt_to_donor_run_params[dapt]
        donor_mtm = _pretrained_to_mtm(donor_params, donor_fishers)

        if ckpt_index is not None and target_param.checkpoint_index != ckpt_index:
            continue

        print(target_param.pretrained_model)
        print(target_mtm.task, donor_params.pretrained_model)

        donor_mtm = donor_mtm.copy(task=target_mtm.task)

        varying_params.append(
            {
                "trial_index": target_param.trial_index,
                "target_ckpt_index": _get_ckpt_index(target_mtm, target_ckpt_summaries),
                "pretrained_mlm_model": donor_params.pretrained_model,
                "models_to_merge": [target_mtm, donor_mtm],
            }
        )

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="6af4ee275e0341329a11ec9d16336e58",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_ROBERTA_LowResource,
        target_fisher_exp=FisherComputation_ROBERTA_TargetTasks,
        donor_fisher_exp=FisherComputation_MlmS2orc,
        mlm_fisher_examples=131072,
    ),
    fixed_params={
        "num_weightings": 76,
        #
        "validation_examples": 2048,
        "sequence_length": 256,
        "batch_size": 128,
        #
        "normalize_fishers": True,
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
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", target_tasks.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class Merge_MlmS2orc_ROBERTA(ExperimentAbc):
    # NOTE: Used 131072 fisher examples.
    pass


@experiment.experiment(
    uuid="8dfedb68dccc4207a749f8c75bc48727",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_ROBERTA_LowResource,
        target_fisher_exp=FisherComputation_ROBERTA_TargetTasks,
        donor_fisher_exp=FisherComputation_MlmS2orc,
        mlm_fisher_examples=131072,
        ckpt_index=9,
    ),
    fixed_params={
        "num_weightings": 76,
        #
        "validation_examples": 2048,
        "sequence_length": 256,
        "batch_size": 128,
        #
        "normalize_fishers": True,
    },
    key_fields={
        "trial_index",
        "models_to_merge",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        #
        scopes.ArgNameBindingSpec("split", "test"),
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
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class Merge_ROBERTA_LastCkpt_TestSet(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="daef5339852f44c386729027c744011a",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_ROBERTA_LowResource,
        target_fisher_exp=FisherComputation_ROBERTA_TargetTasks,
        donor_fisher_exp=FisherComputation_MlmS2orc,
        mlm_fisher_examples=131072,
        ckpt_index=9,
        task_to_dapt_name={
            "chemprot": "allenai/cs_roberta_base",
            "acl_arc": "allenai/biomed_roberta_base",
            "sci_erc": "allenai/biomed_roberta_base",
        },
    ),
    fixed_params={
        "num_weightings": 76,
        #
        "validation_examples": 2048,
        "sequence_length": 256,
        "batch_size": 128,
        #
        "normalize_fishers": True,
    },
    key_fields={
        "trial_index",
        "models_to_merge",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        #
        scopes.ArgNameBindingSpec("split", "test"),
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
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class Merge_ROBERTA_LastCkpt_WrongMerge_TestSet(ExperimentAbc):
    pass
