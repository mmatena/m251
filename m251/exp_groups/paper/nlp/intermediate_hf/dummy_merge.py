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
from m251.fisher.diagonal import dummy_execs
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

from .fisher import (
    FisherComputation_BertBase_HighResource,
    FisherComputation_BertBase_LowResource_LastCkpt,
    FisherComputation_BertBaseFromMnliCkpt_LastCkpt,
    FisherComputation_BertBase_Squad2,
)


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
            "model_merger": dummy_execs.dummy_fisher_model_merger,
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
            #
            #
            "hf_back_compat": False,
            "glue_label_map_overrides": defs.LABEL_MAP_OVERRIDES,
        }

    def get_checkpoint_to_fisher_matrix_uuid(self):
        return {
            m.model_checkpoint_uuid: m.fisher_matrix_uuid for m in self.models_to_merge
        }


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


def create_varying_squad_merge_params(
    exp,
    fisher_exps,
    squad_fisher_exp,
    squad_num_examples=None,
    target_tasks=None,
):
    with exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[f.uuid for f in fisher_exps] + [squad_fisher_exp.uuid]
        )

    run_params, fishers = _get_infos_for_task(exps_data, fisher_exps)
    squad_run_params, squad_fishers = _get_infos_for_task(exps_data, squad_fisher_exp)

    varying_params = []
    for squad_params in squad_run_params:
        if (
            squad_num_examples is not None
            and squad_params.num_examples != squad_num_examples
        ):
            continue
        for target_params in run_params:
            trial_index = 0
            if target_tasks is not None and target_params.task not in target_tasks:
                continue
            if target_params.task not in defs.HIGH_RESOURCE_TASKS:
                trial_index = target_params.trial_index

            target_mtm = _to_mtm(target_params, fishers)
            squad_mtm = _to_mtm(squad_params, squad_fishers)

            # Since we don't need anything with the output of the squad2 model,
            # do this hack to prevent exceptions when loading the model.
            squad_mtm = squad_mtm.copy(task="rte")

            varying_params.append(
                {
                    "trial_index": trial_index,
                    "models_to_merge": [target_mtm, squad_mtm],
                }
            )

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="680a4d14b8b149dabb4be99e3f50d815",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_squad_merge_params,
        fisher_exps=[
            FisherComputation_BertBase_LowResource_LastCkpt,
        ],
        squad_fisher_exp=FisherComputation_BertBase_Squad2,
        squad_num_examples=4096,
        target_tasks={"rte"},
    ),
    fixed_params={
        "pretrained_model": "bert-base-uncased",
        #
        "num_weightings": 51,
        #
        "validation_examples": 2048,
        "sequence_length": 64,
        "batch_size": 512,
        #
        "normalize_fishers": True,
    },
    key_fields={
        "models_to_merge",
        "normalize_fishers",
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
        # This will let us evaluate the downloaded models while not
        # affecting the finetuned ones.
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
    ],
)
class Merge_BertBase_Rte_SquadDonor_4096(ExperimentAbc):
    pass
