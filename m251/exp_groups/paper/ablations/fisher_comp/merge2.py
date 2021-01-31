"""TODO: Add title."""
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.evaluation import eval_execs
from del8.executables.training import optimizers

from m251.data.glue import glue
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.diagonal import variational_diagonal_execs as vardiag_execs
from m251.fisher.diagonal import dummy_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.models.bert import glue_metric_execs as metrics_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import create_pairwise_weightings
from m251.exp_groups.paper.paper_group import ModelToMerge

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .finetune import GlueFinetune
from .fisher import DirectFisherComputation, FisherComputation_Rte_10Epochs


class MergeParamsAbc(ParamsAbc):
    def create_common_bindings(self):
        return {
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


@data_class.data_class()
class DirectMergeParams(MergeParamsAbc):
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
    ):
        pass

    def create_bindings(self):
        return {
            "mergeable_model": diag_execs.diagonal_mergeable_model_from_checkpoint,
            **self.create_common_bindings(),
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


def _filter_run_params_by_task(run_params, task):
    return [run_param for run_param in run_params if run_param.task == task]


def create_varying_params(
    exp,
    target_fisher_exp,
    donor_fisher_exp,
    target_task="rte",
    donor_task="mnli",
    num_fisher_examples=None,
):
    with exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[target_fisher_exp.uuid, donor_fisher_exp.uuid]
        )

    run_params, fishers = _get_infos_for_task(
        exps_data, [target_fisher_exp, donor_fisher_exp]
    )

    target_run_params = _filter_run_params_by_task(run_params, target_task)
    donor_run_params = _filter_run_params_by_task(run_params, donor_task)

    if num_fisher_examples is not None:
        target_run_params = [
            p for p in target_run_params if p.num_examples == num_fisher_examples
        ]
        donor_run_params = [
            p for p in donor_run_params if p.num_examples == num_fisher_examples
        ]

    varying_params = []

    for target_param in target_run_params:
        target_mtm = _finetuned_to_mtm(target_param, fishers)
        for donor_param in donor_run_params:
            donor_mtm = _finetuned_to_mtm(donor_param, fishers)
            varying_params.append(
                {
                    "trial_index": target_param.trial_index,
                    "models_to_merge": [target_mtm, donor_mtm],
                }
            )

    return varying_params


###############################################################################


COMMON_BINDINGS = [
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
]

FIXED_PARAMS = {
    "num_weightings": 51,
    #
    "validation_examples": 2048,
    "sequence_length": 64,
    "batch_size": 2048,
    #
    "pretrained_model": "base",
}

KEY_FIELDS = {
    "trial_index",
    "models_to_merge",
}


###############################################################################


@experiment.experiment(
    uuid="922a351fe2d149d295bf00b2b47fc863",
    group=PaperExpGroup,
    params_cls=DirectMergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        target_fisher_exp=FisherComputation_Rte_10Epochs,
        donor_fisher_exp=DirectFisherComputation,
    ),
    fixed_params=FIXED_PARAMS,
    key_fields=KEY_FIELDS,
    bindings=COMMON_BINDINGS
    + [
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
    ],
)
class MergeDirectFishers(ExperimentAbc):
    pass


###############################################################################
###############################################################################


@data_class.data_class()
class DummyMergeParams(ParamsAbc):
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
    ):
        pass

    def create_bindings(self):
        return {
            "mergeable_model": diag_execs.diagonal_mergeable_model_from_checkpoint,
            "model_merger": dummy_execs.dummy_fisher_model_merger,
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


@experiment.experiment(
    uuid="08692247e1a54d17b946dcc3d63463cf",
    group=PaperExpGroup,
    params_cls=DirectMergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        target_fisher_exp=FisherComputation_Rte_10Epochs,
        donor_fisher_exp=DirectFisherComputation,
        num_fisher_examples=256,
    ),
    fixed_params=FIXED_PARAMS,
    key_fields=KEY_FIELDS,
    bindings=COMMON_BINDINGS
    + [
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
    ],
)
class MergeDirectFishers_Dummy(ExperimentAbc):
    pass
