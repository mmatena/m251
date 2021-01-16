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

from . import defs

from .finetune import GlueFinetune
from .fisher import (
    FisherComputation_LastCkpt,
    FisherComputation_AltCkpts,
    FisherComputation_LowRegStrength,
)


HIGH_RESOURCE_TASKS = defs.HIGH_RESOURCE_TASKS
HIGH_RESOURCE_TRIALS = defs.HIGH_RESOURCE_TRIALS

LOW_RESOURCE_TASKS = defs.LOW_RESOURCE_TASKS


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
            "mergeable_model": diag_execs.diagonal_mergeable_model_from_checkpoint,
            "model_merger": diag_execs.diagonal_model_merger,
            #
            "checkpoint_to_fisher_matrix_uuid": self.get_checkpoint_to_fisher_matrix_uuid(),
            "weightings": create_pairwise_weightings(self.num_weightings),
            #
            "checkpoints": [m.model_checkpoint_uuid for m in self.models_to_merge],
            "checkpoint_tasks": [m.task for m in self.models_to_merge],
            #
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


def create_varying_params(
    exp,
    fisher_exps,
):
    assert HIGH_RESOURCE_TRIALS == 1
    with exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[f.uuid for f in fisher_exps]
        )

    run_params, fishers = _get_infos_for_task(exps_data, fisher_exps)

    varying_params = []
    for p1, p2 in itertools.combinations(run_params, 2):
        if (
            p1.trial_index != p2.trial_index
            and p1.task not in HIGH_RESOURCE_TASKS
            and p2.task not in HIGH_RESOURCE_TASKS
        ):
            continue
        elif p1.task == p2.task:
            continue

        trial_index = 0
        if p1.task not in HIGH_RESOURCE_TASKS:
            trial_index = p1.trial_index
        elif p2.task not in HIGH_RESOURCE_TASKS:
            trial_index = p2.trial_index

        mtm1 = _finetuned_to_mtm(p1, fishers)
        mtm2 = _finetuned_to_mtm(p2, fishers)

        varying_params.append(
            {
                "trial_index": trial_index,
                "models_to_merge": [mtm1, mtm2],
            }
        )
        varying_params.append(
            {
                "trial_index": f"rev_{trial_index}",
                "models_to_merge": [mtm2, mtm1],
            }
        )
    return varying_params


# def create_varying_params(
#     exp,
#     fisher_exp,
# ):
#     assert HIGH_RESOURCE_TRIALS == 1
#     with fisher_exp.get_storage() as storage:
#         exps_data = storage.retrieve_storage_data(
#             experiment_uuid=[fisher_exp.uuid])

#     run_params, fishers = _get_infos_for_task(exps_data, fisher_exp)

#     varying_params = []
#     for p1, p2 in itertools.combinations(run_params, 2):
#         if p1.trial_index != p2.trial_index and p1.task not in HIGH_RESOURCE_TASKS and p2.task not in HIGH_RESOURCE_TASKS:
#             continue
#         elif p1.task == p2.task:
#             continue

#         trial_index = 0
#         if p1.task not in HIGH_RESOURCE_TASKS:
#             trial_index = p1.trial_index
#         elif p2.task not in HIGH_RESOURCE_TASKS:
#             trial_index = p2.trial_index

#         mtm1 = _finetuned_to_mtm(p1, fishers)
#         mtm2 = _finetuned_to_mtm(p2, fishers)

#         varying_params.append(
#             {
#                 "trial_index": trial_index,
#                 "models_to_merge": [mtm1, mtm2],
#             }
#         )
#         varying_params.append(
#             {
#                 "trial_index": f'rev_{trial_index}',
#                 "models_to_merge": [mtm2, mtm1],
#             }
#         )
#     return varying_params


###############################################################################


@experiment.experiment(
    uuid="f7a06ea1593e415e9b8e9192fe36d9f3",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
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
        "num_weightings": 51,
        #
        "validation_examples": 2048,
        "sequence_length": 64,
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
class Merge_Pairs_Normalized_LastCkpt(ExperimentAbc):
    pass
