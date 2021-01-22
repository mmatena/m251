"""TODO: Add title."""
import itertools
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

from . import defs

from .fisher_large import FisherComputation_RobertLargeMnli_Rte_LastCkpt


@data_class.data_class()
class MergeParams(ParamsAbc):
    def __init__(
        self,
        #
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
            "single_score_from_results": merging_execs.single_score_from_results,
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
            "task": self.models_to_merge[0].task,
            "tasks": [self.models_to_merge[0].task],
            #
            "pretrained_model": self.pretrained_model,
            #
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            #
            "normalize_fishers": self.normalize_fishers,
            "multitask_merge": False,
        }

    def get_checkpoint_to_fisher_matrix_uuid(self):
        return {
            m.model_checkpoint_uuid: m.fisher_matrix_uuid for m in self.models_to_merge
        }

    def create_preload_blob_uuids(self):
        dikt = self.get_checkpoint_to_fisher_matrix_uuid()
        return tuple(set(dikt.keys()) | set(dikt.values()))


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


def _rotate(lst, n):
    return tuple(lst[n:] + lst[:n])


def create_varying_params_single_task(exp, fisher_exp):
    with exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[fisher_exp.uuid])

    run_params, fishers = _get_infos_for_task(exps_data, [fisher_exp])

    mtms = []
    for params in run_params:
        mtm = _finetuned_to_mtm(params, fishers)
        mtms.append(mtm)

    varying_params = []
    for i in range(len(mtms)):
        varying_params.append({"models_to_merge": _rotate(mtms, i)})

    return varying_params


@experiment.experiment(
    uuid="d0a7e86c96f74755ba28b2a83df2af79",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_weighting_search_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params_single_task,
        fisher_exp=FisherComputation_RobertLargeMnli_Rte_LastCkpt,
    ),
    fixed_params={
        "pretrained_model": "roberta-large-mnli",
        #
        "search_steps": 250,
        "search_num_inits": 3,
        "search_num_examples": 512,
        "final_evaluate_num_examples": 2048,
        #
        "sequence_length": 64,
        "batch_size": 512,
        #
        "normalize_fishers": True,
    },
    key_fields={
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
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("glue_label_map_overrides", defs.LABEL_MAP_OVERRIDES),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class Merge_RteFromMmnli_RobertaLarge(ExperimentAbc):
    pass
