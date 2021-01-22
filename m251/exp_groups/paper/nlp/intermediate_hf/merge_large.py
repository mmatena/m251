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
from ..intermediate.fisher import FisherComputation_LastCkpt


BAD_FINETUNE_RUN_UUIDS = frozenset(
    {
        "37dbf11090b047b2ba2e9996597e22ab",
        "ab6ce15a17ad4ea287c08093270ee494",
        "b8103c8e19054604a420b1ec2c1e4a15",
        "2b23839254934890acd9fab09803382c",
    }
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
            "mergeable_model": diag_execs.diagonal_mergeable_model_from_checkpoint,
            "model_merger": diag_execs.diagonal_model_merger,
            #
            "checkpoint_to_fisher_matrix_uuid": self.get_checkpoint_to_fisher_matrix_uuid(),
            "weightings": create_pairwise_weightings(self.num_weightings),
            #
            "checkpoints": [m.model_checkpoint_uuid for m in self.models_to_merge],
            "checkpoint_tasks": [m.task for m in self.models_to_merge],
            "additional_model_bindings": [
                m.additional_model_bindings for m in self.models_to_merge
            ],
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


def _finetuned_to_mtm(run_params, fishers, additional_model_bindings):
    return ModelToMerge(
        task=run_params.task,
        train_run_uuid=run_params.finetuned_run_uuid,
        fisher_run_uuid=run_params.run_uuid,
        model_checkpoint_uuid=run_params.finetuned_ckpt_uuid,
        fisher_matrix_uuid=fishers[run_params.run_uuid],
        additional_model_bindings=additional_model_bindings,
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
    additional_target_model_bindings,
    additional_donor_model_bindings,
):
    with exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[target_fisher_exp.uuid, donor_fisher_exp.uuid]
        )

    target_run_params, target_fishers = _get_infos_for_task(
        exps_data, [target_fisher_exp]
    )
    donor_run_params, donor_fishers = _get_infos_for_task(exps_data, [donor_fisher_exp])

    varying_params = []
    for target_param in target_run_params:
        for donor_param in donor_run_params:
            if donor_param.finetuned_run_uuid in BAD_FINETUNE_RUN_UUIDS:
                continue
            elif (
                target_param.trial_index != donor_param.trial_index
                and donor_param.task not in defs.HIGH_RESOURCE_TASKS
            ):
                continue
            mtm1 = _finetuned_to_mtm(
                target_param, target_fishers, additional_target_model_bindings
            )
            mtm2 = _finetuned_to_mtm(
                donor_param, donor_fishers, additional_donor_model_bindings
            )
            varying_params.append(
                {
                    "trial_index": target_param.trial_index,
                    "models_to_merge": [mtm1, mtm2],
                }
            )
    return varying_params


@experiment.experiment(
    uuid="cb64294006334979a53a2534e1d04a1f",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        target_fisher_exp=FisherComputation_RobertLargeMnli_Rte_LastCkpt,
        donor_fisher_exp=FisherComputation_LastCkpt,
        additional_target_model_bindings=[
            ["pretrained_model", "roberta-large-mnli"],
            ["hf_back_compat", False],
            ["pretrained_body_only", True],
            ["use_roberta_head", True],
        ],
        additional_donor_model_bindings=[
            ["pretrained_model", "roberta-large"],
            ["hf_back_compat", True],
            ["pretrained_body_only", True],
            ["use_roberta_head", False],
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
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("glue_label_map_overrides", defs.LABEL_MAP_OVERRIDES),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class Merge_Pairs_Normalized_LastCkpt(ExperimentAbc):
    pass
