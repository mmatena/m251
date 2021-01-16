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

from ..intermediate import defs

from ..intermediate.finetune import GlueFinetune
from ..intermediate.fisher import (
    FisherComputation_LastCkpt,
    FisherComputation_AltCkpts,
    FisherComputation_LowRegStrength,
)


HIGH_RESOURCE_TASKS = defs.HIGH_RESOURCE_TASKS
HIGH_RESOURCE_TRIALS = defs.HIGH_RESOURCE_TRIALS

LOW_RESOURCE_TASKS = defs.LOW_RESOURCE_TASKS

BAD_FINETUNE_RUN_UUIDS = defs.BAD_FINETUNE_RUN_UUIDS


@data_class.data_class()
class MergeParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        models_to_merge,
        #
        search_steps,
        search_num_inits,
        search_num_examples,
        final_evaluate_num_examples,
        #
        sequence_length,
        batch_size,
        #
        pretrained_model,
        #
        normalize_fishers,
    ):
        pass

    def create_bindings(self):
        return {
            "mergeable_model": diag_execs.diagonal_mergeable_model_from_checkpoint,
            "model_merger": diag_execs.diagonal_model_merger,
            "merge_weighting_searcher": diag_execs.diagonal_model_merge_weighting_search,
            #
            # This scores by average score across tasks.
            "single_score_from_results": merging_execs.average_score_from_results,
            #
            "merge_weighting_search_steps": self.search_steps,
            "merge_weighting_num_inits": self.search_num_inits,
            "search_num_examples": self.search_num_examples,
            "final_evaluate_num_examples": self.final_evaluate_num_examples,
            #
            "checkpoint_to_fisher_matrix_uuid": self.get_checkpoint_to_fisher_matrix_uuid(),
            #
            "checkpoints": [m.model_checkpoint_uuid for m in self.models_to_merge],
            "checkpoint_tasks": [m.task for m in self.models_to_merge],
            #
            "tasks": [m.task for m in self.models_to_merge],
            #
            "pretrained_model": self.pretrained_model,
            #
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            #
            "normalize_fishers": self.normalize_fishers,
            "multitask_merge": True,
        }

    def get_checkpoint_to_fisher_matrix_uuid(self):
        return {
            m.model_checkpoint_uuid: m.fisher_matrix_uuid for m in self.models_to_merge
        }

    def create_preload_blob_uuids(self):
        dikt = self.get_checkpoint_to_fisher_matrix_uuid()
        return tuple(set(dikt.keys()) | set(dikt.values()))


###############################################################################


def _finetuned_to_mtm(run_params, fishers):
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


def create_varying_params(exp, fisher_exps, bad_run_uuids=BAD_FINETUNE_RUN_UUIDS):
    assert HIGH_RESOURCE_TRIALS == 1
    with exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[f.uuid for f in fisher_exps]
        )

    run_params, fishers = _get_infos_for_task(exps_data, fisher_exps)
    run_params = [p for p in run_params if p.run_uuid not in bad_run_uuids]

    task_to_params = collections.defaultdict(dict)
    for p in run_params:
        task_to_params[p.task][p.trial_index] = p

    tasks = sorted(task_to_params.keys())

    num_indices = -1
    for task, params in task_to_params.items():
        default_index = min(params.keys())
        num_indices = max(num_indices, max(params.keys()))
        params["default"] = params[default_index]

    varying_params = []
    for trial_index in range(num_indices + 1):
        mtms = []
        for task in tasks:
            if trial_index in task_to_params[task]:
                fisher_params = task_to_params[task][trial_index]
            else:
                fisher_params = task_to_params[task]["default"]
            mtms.append(_finetuned_to_mtm(fisher_params, fishers))

        #
        #
        #
        #
        #
        # mtms = mtms[:2]
        #
        #
        #
        #
        #
        varying_param = {
            "trial_index": trial_index,
            "models_to_merge": tuple(mtms),
        }
        varying_params.append(varying_param)

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="2d69b52db90a431ca27ca34de2e7958b",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_weighting_search_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        fisher_exps=[
            FisherComputation_LastCkpt,
            FisherComputation_AltCkpts,
            FisherComputation_LowRegStrength,
        ],
    ),
    fixed_params={
        "pretrained_model": "roberta-large",
        #
        "search_steps": 250,
        "search_num_inits": 3,
        "search_num_examples": 512,
        "final_evaluate_num_examples": 2048,
        #
        "sequence_length": 64,
        # "batch_size": 128,
        "batch_size": 512,
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
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", glue.glue_robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("merge_on_cpu", True),
    ],
)
class Merge_Most(ExperimentAbc):
    pass
