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
from m251.fisher.diagonal import variational_diagonal_execs as vardiag_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.models.bert import glue_metric_execs as metrics_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import create_pairwise_weightings
from m251.exp_groups.paper.paper_group import ModelToMerge

from m251.exp_groups.paper.paper_group import PaperExpGroup

from m251.exp_groups.paper.results import utils as result_utils

from .finetune import Finetune_LowResource
from .fisher import (
    FisherComputation_MlmTargetTask,
    FisherComputation_TargetTask,
    FisherComputation_MlmS2orc,
)
from .eval import Eval_LowResource_NoDapt_All

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


COMMON_BINDINGS = [
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
]

COMMON_FIXED_PARAMS = {
    "num_weightings": 76,
    #
    "validation_examples": 2048,
    "sequence_length": 256,
    "batch_size": 128,
    #
    "normalize_fishers": True,
}

KEY_FIELDS = {
    "trial_index",
    "models_to_merge",
}

COMMON_EXP_KWARGS = {
    "group": PaperExpGroup,
    "params_cls": MergeParams,
    "fixed_params": COMMON_FIXED_PARAMS,
    "key_fields": KEY_FIELDS,
    "bindings": COMMON_BINDINGS,
    "executable_cls": merging_execs.merge_and_evaluate_from_checkpoints,
}


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
    target_pretrained_model="roberta-base",
    target_ckpt_index=None,
    target_task=None,
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
    target_run_params = [
        p for p in target_run_params if p.pretrained_model == target_pretrained_model
    ]
    target_ckpt_summaries = _get_ckpt_summaries(exps_data, train_exp)

    donor_run_params, donor_fishers = _get_infos_for_task(exps_data, donor_fisher_exp)
    task_to_donor_run_params = {p.task: p for p in donor_run_params}

    varying_params = []

    for target_param in target_run_params:
        target_mtm = _finetuned_to_mtm(target_param, target_fishers)
        if target_task is not None and target_mtm.task != target_task:
            continue
        donor_params = task_to_donor_run_params[target_mtm.task]
        donor_mtm = _pretrained_to_mtm(donor_params, donor_fishers)
        varying_params.append(
            {
                "trial_index": target_param.trial_index,
                "target_ckpt_index": _get_ckpt_index(target_mtm, target_ckpt_summaries),
                "pretrained_mlm_model": donor_params.pretrained_model,
                "models_to_merge": [target_mtm, donor_mtm],
            }
        )

    if target_ckpt_index is not None:
        varying_params = [
            v for v in varying_params if v["target_ckpt_index"] == target_ckpt_index
        ]

    return varying_params


###############################################################################

# NOTE: The MLM Fisher had a mucher higher magnitude than the target task fisher,
# which led to poor performance in this experiment.
@experiment.experiment(
    uuid="1582aef382d14d64b1f2ecc6955becc1",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_LowResource,
        target_fisher_exp=FisherComputation_TargetTask,
        donor_fisher_exp=FisherComputation_MlmTargetTask,
    ),
    fixed_params={
        "num_weightings": 51,
        #
        "validation_examples": 2048,
        "sequence_length": 256,
        "batch_size": 128,
        #
        "normalize_fishers": False,
    },
    key_fields=KEY_FIELDS,
    bindings=COMMON_BINDINGS,
)
class Merge_MlmTargetTask(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="46fa2152a0334473a92ae95f775c6a79",
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_LowResource,
        target_fisher_exp=FisherComputation_TargetTask,
        donor_fisher_exp=FisherComputation_MlmTargetTask,
        target_ckpt_index=9,
    ),
    **COMMON_EXP_KWARGS,
)
class Merge_MlmTargetTask_Normalized_LastCkpt(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="ae6cf6d35eba4a689ceb76d1fc5ee40a",
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_LowResource,
        target_fisher_exp=FisherComputation_TargetTask,
        donor_fisher_exp=FisherComputation_MlmTargetTask,
        target_ckpt_index=8,
        target_task="acl_arc",
    ),
    **COMMON_EXP_KWARGS,
)
class Merge_MlmTargetTask_Normalized_AclArc_BestCkpt(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="f20fba93aea7413ea8760a6cad327166",
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_LowResource,
        target_fisher_exp=FisherComputation_TargetTask,
        donor_fisher_exp=FisherComputation_MlmTargetTask,
        target_ckpt_index=8,
        target_task="sci_erc",
    ),
    **COMMON_EXP_KWARGS,
)
class Merge_MlmTargetTask_Normalized_SciErc_Ckpt8(ExperimentAbc):
    pass


###############################################################################
###############################################################################


def create_varying_params_s2orc(
    exp,
    train_exp,
    target_fisher_exp,
    donor_fisher_exp,
    target_pretrained_model="roberta-base",
    target_ckpt_index=None,
    target_task=None,
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
    target_run_params = [
        p for p in target_run_params if p.pretrained_model == target_pretrained_model
    ]
    target_ckpt_summaries = _get_ckpt_summaries(exps_data, train_exp)

    donor_run_params, donor_fishers = _get_infos_for_task(exps_data, donor_fisher_exp)
    dapt_to_donor_run_params = {p.pretrained_model: p for p in donor_run_params}

    varying_params = []

    for target_param in target_run_params:
        target_mtm = _finetuned_to_mtm(target_param, target_fishers)
        if target_task is not None and target_mtm.task != target_task:
            continue
        dapt = TASK_TO_DAPT_NAME[target_mtm.task]
        donor_params = dapt_to_donor_run_params[dapt]
        donor_mtm = _pretrained_to_mtm(donor_params, donor_fishers)
        donor_mtm = donor_mtm.copy(task=target_mtm.task)
        varying_params.append(
            {
                "trial_index": target_param.trial_index,
                "target_ckpt_index": _get_ckpt_index(target_mtm, target_ckpt_summaries),
                "pretrained_mlm_model": donor_params.pretrained_model,
                "models_to_merge": [target_mtm, donor_mtm],
            }
        )

    if target_ckpt_index is not None:
        varying_params = [
            v for v in varying_params if v["target_ckpt_index"] == target_ckpt_index
        ]

    return varying_params


@experiment.experiment(
    uuid="a0b5458f6dbf474cbeccfe4e948f4ba8",
    varying_params=functools.partial(
        create_varying_params_s2orc,
        train_exp=Finetune_LowResource,
        target_fisher_exp=FisherComputation_TargetTask,
        donor_fisher_exp=FisherComputation_MlmS2orc,
        target_ckpt_index=9,
    ),
    **COMMON_EXP_KWARGS,
)
class Merge_MlmS2orc_Normalized_LastCkpt(ExperimentAbc):
    pass


###############################################################################
###############################################################################


def _get_best_eval_ckpt_uuids(exps_data, eval_exp):
    besties = set()
    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=eval_exp.uuid)
    for run_id in run_ids:
        eval_run = exps_data.get_run_data(run_id)

        res = eval_run.get_items_by_class(eval_execs.CheckpointEvaluationResults)
        best = max(res, key=lambda r: get_single_score(r.results))
        besties.add(best.checkpoint_blob_uuid)
    return besties


def create_varying_params_s2orc_best(
    exp,
    train_exp,
    eval_exp,
    target_fisher_exp,
    donor_fisher_exp,
    target_pretrained_model="roberta-base",
    target_task=None,
):
    with target_fisher_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[
                train_exp.uuid,
                eval_exp.uuid,
                target_fisher_exp.uuid,
                donor_fisher_exp.uuid,
            ]
        )

    target_run_params, target_fishers = _get_infos_for_task(
        exps_data, target_fisher_exp
    )
    target_run_params = [
        p for p in target_run_params if p.pretrained_model == target_pretrained_model
    ]
    target_ckpt_summaries = _get_ckpt_summaries(exps_data, train_exp)

    donor_run_params, donor_fishers = _get_infos_for_task(exps_data, donor_fisher_exp)
    dapt_to_donor_run_params = {p.pretrained_model: p for p in donor_run_params}

    ckpt_uuids = _get_best_eval_ckpt_uuids(exps_data, eval_exp)

    varying_params = []

    for target_param in target_run_params:
        target_mtm = _finetuned_to_mtm(target_param, target_fishers)
        if target_task is not None and target_mtm.task != target_task:
            continue
        elif target_mtm.model_checkpoint_uuid not in ckpt_uuids:
            continue
        dapt = TASK_TO_DAPT_NAME[target_mtm.task]
        donor_params = dapt_to_donor_run_params[dapt]
        donor_mtm = _pretrained_to_mtm(donor_params, donor_fishers)
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


@experiment.experiment(
    uuid="8d9d214d970a46caa0940073523f6cf7",
    varying_params=functools.partial(
        create_varying_params_s2orc_best,
        train_exp=Finetune_LowResource,
        eval_exp=Eval_LowResource_NoDapt_All,
        target_fisher_exp=FisherComputation_TargetTask,
        donor_fisher_exp=FisherComputation_MlmS2orc,
    ),
    **COMMON_EXP_KWARGS,
)
class Merge_MlmS2orc_Normalized_BestCkpt(ExperimentAbc):
    pass
