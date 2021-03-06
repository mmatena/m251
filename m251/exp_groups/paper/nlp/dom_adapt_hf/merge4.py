"""TODO: Add title."""
import collections
import functools
import itertools

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
from m251.data.domains import target_tasks

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

from .fisher2 import FisherComputation_ROBERTA_TargetTasks_LastCkpt_AllVars
from .fisher2 import FisherComputation_DAPT_HeadOnly_TargetTasks_LastCkpt_AllVars


TASK_TO_DAPT_NAME = target_tasks.TASK_TO_DAPT_NAME

get_single_score = result_utils.get_single_score


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
        #
        normalize_fishers,
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
            #
            "normalize_fishers": self.normalize_fishers,
        }

    def get_checkpoint_to_fisher_matrix_uuid(self):
        return {
            m.model_checkpoint_uuid: m.fisher_matrix_uuid for m in self.models_to_merge
        }

    def create_preload_blob_uuids(self):
        dikt = self.get_checkpoint_to_fisher_matrix_uuid()
        return tuple(set(dikt.keys()) | set(dikt.values()))


###############################################################################


###############################################################################


def _to_mtm(run_params, fishers):
    # Supports both fine-tuned and downloaded models.
    train_run_uuid = getattr(run_params, "finetuned_run_uuid", None)
    model_checkpoint_uuid = getattr(
        run_params, "finetuned_ckpt_uuid", run_params.pretrained_model
    )
    return ModelToMerge(
        task=run_params.task,
        train_run_uuid=train_run_uuid,
        fisher_run_uuid=run_params.run_uuid,
        model_checkpoint_uuid=model_checkpoint_uuid,
        fisher_matrix_uuid=fishers[run_params.run_uuid],
    )


def _map_run_uuid_to_fisher_matrix_uuid(run_uuids, run_datas):
    return {
        run_uuid: run_data.get_single_item_by_class(
            fisher_execs.SavedFisherMatrix
        ).blob_uuid
        for run_uuid, run_data in zip(run_uuids, run_datas)
    }


def _get_infos_for_task(exps_data, fisher_exp):
    if isinstance(fisher_exp, (list, tuple, set, frozenset)):
        run_params, fishers = [], {}
        for exp in fisher_exp:
            rp, f = _get_infos_for_task(exps_data, exp)
            run_params.extend(rp)
            fishers.update(f)
        return run_params, fishers

    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=fisher_exp.uuid)
    run_datas = [exps_data.get_run_data(run_id) for run_id in run_ids]

    run_params = [
        run_data.get_single_item_by_class(fisher_exp.params_cls)
        for run_data in run_datas
    ]

    fishers = _map_run_uuid_to_fisher_matrix_uuid(run_ids, run_datas)

    return run_params, fishers


def create_varying_params(
    exp,
    target_fisher_exp,
    donor_fisher_exp,
):
    with exp.get_storage() as storage:
        target_exps_data = storage.retrieve_storage_data(
            experiment_uuid=[target_fisher_exp.uuid]
        )
        donor_exps_data = storage.retrieve_storage_data(
            experiment_uuid=[donor_fisher_exp.uuid]
        )

    target_run_params, target_fishers = _get_infos_for_task(
        target_exps_data, target_fisher_exp
    )
    donor_run_params, donor_fishers = _get_infos_for_task(
        donor_exps_data, donor_fisher_exp
    )

    varying_params = []
    for target_param in target_run_params:
        for donor_param in donor_run_params:
            if target_param.task != donor_param.task:
                continue
            elif target_param.trial_index != donor_param.trial_index:
                continue
            mtm1 = _to_mtm(target_param, target_fishers)
            mtm2 = _to_mtm(donor_param, donor_fishers)
            varying_params.append(
                {
                    "trial_index": target_param.trial_index,
                    "models_to_merge": [mtm1, mtm2],
                }
            )
    return varying_params


###############################################################################


@experiment.experiment(
    uuid="3def6939c860465da94813cb5fbf0ff2",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        target_fisher_exp=FisherComputation_ROBERTA_TargetTasks_LastCkpt_AllVars,
        donor_fisher_exp=FisherComputation_DAPT_HeadOnly_TargetTasks_LastCkpt_AllVars,
    ),
    fixed_params={
        "num_weightings": 76,
        #
        "validation_examples": 2048,
        "sequence_length": 256,
        "batch_size": 128,
        #
        "normalize_fishers": True,
        #
        "pretrained_model": "roberta-base",
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
        #
        scopes.ArgNameBindingSpec("all_variables_mergeable", True),
        #
        scopes.ArgNameBindingSpec("min_fisher", 1e-20),
    ],
)
class Merge_ROBERTA_LastCkpt_TestSet_WithDaptAllVars(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="b7e9f48636354bd0b0911da070943cae",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        target_fisher_exp=FisherComputation_ROBERTA_TargetTasks_LastCkpt_AllVars,
        donor_fisher_exp=FisherComputation_DAPT_HeadOnly_TargetTasks_LastCkpt_AllVars,
    ),
    fixed_params={
        "num_weightings": 76,
        #
        "validation_examples": 2048,
        "sequence_length": 256,
        "batch_size": 128,
        #
        "normalize_fishers": True,
        #
        "pretrained_model": "roberta-base",
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
        #
        scopes.ArgNameBindingSpec("all_variables_mergeable", False),
        #
        scopes.ArgNameBindingSpec("min_fisher", 1e-20),
    ],
)
class Merge_ROBERTA_LastCkpt_TestSet_WithDaptAllVars_MergeOnlyBody(ExperimentAbc):
    pass
