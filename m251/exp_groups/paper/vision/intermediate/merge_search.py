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

from m251.data.image import image_classification
from m251.models import model_execs
from m251.models.simclr import simclr
from m251.models.simclr import simclr_classifier_execs as sc_exe

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

from . import defs

from .fisher import FisherComputation_1x_Exact, FisherComputation_4x_Sampling


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
        image_size,
        batch_size,
        #
        pretrained_model,
        #
        normalize_fishers,
        #
        trial_index=0,
    ):
        pass

    def create_bindings(self):
        return {
            "mergeable_model": diag_execs.diagonal_mergeable_model_from_checkpoint_or_pretrained,
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
            "image_size": self.image_size,
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
        return tuple(set(dikt.keys()))


###############################################################################


def _pretrained_to_mtm(run_params, fishers):
    return ModelToMerge(
        task=run_params.task,
        train_run_uuid=None,
        fisher_run_uuid=run_params.run_uuid,
        model_checkpoint_uuid=run_params.pretrained_model,
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
    fisher_exps,
):
    with exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[f.uuid for f in fisher_exps]
        )

    run_params, fishers = _get_infos_for_task(exps_data, fisher_exps)

    #
    #
    for p in run_params:
        print(p.task)
    #
    #

    varying_params = []
    for p1, p2 in itertools.combinations(run_params, 2):
        if p1.trial_index != p2.trial_index or p1.task == p2.task:
            continue

        trial_index = p1.trial_index

        mtm1 = _pretrained_to_mtm(p1, fishers)
        mtm2 = _pretrained_to_mtm(p2, fishers)

        varying_params.append(
            {
                "pretrained_model": p1.pretrained_model,
                "trial_index": trial_index,
                "models_to_merge": [mtm1, mtm2],
            }
        )
        varying_params.append(
            {
                "pretrained_model": p2.pretrained_model,
                "trial_index": trial_index,
                "models_to_merge": [mtm2, mtm1],
            }
        )

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="4cafeea5b54341be8b94299ac5dd87e0",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_weighting_search_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        fisher_exps=[FisherComputation_1x_Exact],
    ),
    fixed_params={
        "search_steps": 128,
        "search_num_inits": 1,
        "search_num_examples": 2048,
        "final_evaluate_num_examples": 2048,
        #
        "image_size": simclr.IMAGE_SIZE,
        "batch_size": 256,
        #
        "normalize_fishers": True,
    },
    key_fields={
        "models_to_merge",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("multitask_merge", False),
        #
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", image_classification.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("initializer", sc_exe.simclr_initializer),
        scopes.ArgNameBindingSpec("loader", sc_exe.simclr_loader),
        scopes.ArgNameBindingSpec("builder", sc_exe.simclr_builder),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
        #
        scopes.ArgNameBindingSpec("color_distort", False),
        scopes.ArgNameBindingSpec("test_crop", True),
        #
        scopes.ArgNameBindingSpec("merge_on_cpu", False),
    ],
)
class Merge_1x_Exact(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="c9f833022a624c8aa6349f4f1f49e177",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_weighting_search_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        fisher_exps=[FisherComputation_4x_Sampling],
    ),
    fixed_params={
        "search_steps": 128,
        "search_num_inits": 1,
        "search_num_examples": 2048,
        "final_evaluate_num_examples": 2048,
        #
        "image_size": simclr.IMAGE_SIZE,
        "batch_size": 256,
        #
        "normalize_fishers": True,
    },
    key_fields={
        "models_to_merge",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("multitask_merge", False),
        #
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", image_classification.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("initializer", sc_exe.simclr_initializer),
        scopes.ArgNameBindingSpec("loader", sc_exe.simclr_loader),
        scopes.ArgNameBindingSpec("builder", sc_exe.simclr_builder),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
        #
        scopes.ArgNameBindingSpec("color_distort", False),
        scopes.ArgNameBindingSpec("test_crop", True),
        #
        scopes.ArgNameBindingSpec("merge_on_cpu", True),
    ],
)
class Merge_4x(ExperimentAbc):
    pass
