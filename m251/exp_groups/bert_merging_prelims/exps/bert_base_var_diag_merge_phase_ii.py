"""TODO: Add title."""
import collections
import itertools
import functools

from absl import logging

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.data import tfds as tfds_execs
from del8.executables.evaluation import eval_execs
from del8.executables.models import checkpoints as ckpt_exec
from del8.executables.training import fitting
from del8.executables.training import optimizers

from m251.data.glue import glue
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

from m251.fisher.diagonal import diagonal_execs
from m251.fisher.diagonal import variational_diagonal_execs as vardiag_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.models import model_execs
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import glue_metric_execs as metrics_exe

from .diag_merge_glue_iso import ModelToMerge
from ..group import BertMergingPrelimsGroup

from . import finetune_bert_base
from . import bert_base_fisher_var_diag


def create_pair_weightings(num_weightings):
    denom = num_weightings + 1
    return [((i + 1) / denom, 1 - (i + 1) / denom) for i in range(num_weightings)]


@data_class.data_class()
class MergePairParams(object):
    def __init__(
        self,
        #
        models_to_merge,
        num_weightings,
        #
        fisher_type,
        #
        sequence_length,
        batch_size,
        validation_examples,
        #
        pretrained_model,
        task,
        reg_strength,
        reg_type,
    ):
        pass

    def get_checkpoint_to_fisher_matrix_uuid(self):
        return {
            m.model_checkpoint_uuid: m.fisher_matrix_uuid for m in self.models_to_merge
        }

    def create_binding_specs(self):
        if self.fisher_type == "variational_diagonal":
            fisher_bindings = [
                scopes.ArgNameBindingSpec(
                    "mergeable_model",
                    vardiag_execs.diagonal_mergeable_model_from_checkpoint,
                ),
                scopes.ArgNameBindingSpec(
                    "model_merger", diagonal_execs.diagonal_model_merger
                ),
            ]
        else:
            raise ValueError(f"Invalid fisher_type {self.fisher_type}.")

        return [
            scopes.ArgNameBindingSpec(
                "checkpoint_to_fisher_matrix_uuid",
                self.get_checkpoint_to_fisher_matrix_uuid(),
            ),
            scopes.ArgNameBindingSpec(
                "weightings", create_pair_weightings(self.num_weightings)
            ),
            #
            scopes.ArgNameBindingSpec(
                "checkpoints", [m.model_checkpoint_uuid for m in self.models_to_merge]
            ),
            scopes.ArgNameBindingSpec(
                "checkpoint_tasks", [m.task for m in self.models_to_merge]
            ),
            #
            scopes.ArgNameBindingSpec("task", self.models_to_merge[0].task),
            scopes.ArgNameBindingSpec("tasks", [m.task for m in self.models_to_merge]),
            #
            scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
            #
            scopes.ArgNameBindingSpec("num_examples", self.validation_examples),
            scopes.ArgNameBindingSpec("sequence_length", self.sequence_length),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
        ] + fisher_bindings

    def create_preload_blob_uuids(self):
        dikt = self.get_checkpoint_to_fisher_matrix_uuid()
        return tuple(set(dikt.keys()) | set(dikt.values()))


@experiment.with_experiment_storages()
def create_varying_params(exp, train_exp, fisher_exp, target_task, donor_task):
    train_run_uuids = train_exp.retrieve_run_uuids(RunState.FINISHED)
    fisher_run_uuids = fisher_exp.retrieve_run_uuids(RunState.FINISHED)

    train_run_params = {
        rid: train_exp.retrieve_run_params(rid) for rid in train_run_uuids
    }
    fisher_run_params = [
        fisher_exp.retrieve_run_params(rid) for rid in fisher_run_uuids
    ]

    target_fisher_run_uuids = [
        uuid
        for uuid, params in zip(fisher_run_uuids, fisher_run_params)
        if params.task == target_task
    ]
    donor_fisher_run_uuids = [
        uuid
        for uuid, params in zip(fisher_run_uuids, fisher_run_params)
        if params.task == donor_task
    ]

    grouping_to_params = collections.defaultdict(list)
    for fi_rid, fi_rp in zip(fisher_run_uuids, fisher_run_params):
        assert fi_rp.finetuned_exp_uuid == train_exp.uuid

        tr_rp = train_run_params[fi_rp.finetuned_run_uuid]

        # NOTE: We could do this in fewer db calls and probably be faster.
        summary = fisher_exp.retrieve_single_item_by_class(
            fisher_execs.FisherMatricesSummary, fi_rid
        )
        for i, sfm_uuid in enumerate(summary.saved_fisher_matrix_uuids):
            saved_fisher_matrix = fisher_exp.get_storage().retrieve_item(sfm_uuid)

            grouping_key = {
                "index": i,
                "variational_fisher_beta": fi_rp.variational_fisher_beta,
                "learning_rate": fi_rp.learning_rate,
                # "num_examples": fi_rp.num_examples,
                #
                "pretrained_model": fi_rp.pretrained_model,
                "reg_strength": tr_rp.reg_strength,
                "reg_type": tr_rp.reg_type,
            }
            grouping_key = hashabledict(grouping_key)

            model_to_merge = ModelToMerge(
                task=tr_rp.task,
                train_run_uuid=fi_rp.finetuned_run_uuid,
                fisher_run_uuid=fi_rid,
                model_checkpoint_uuid=fi_rp.finetuned_ckpt_uuid,
                fisher_matrix_uuid=saved_fisher_matrix.blob_uuid,
            )

            grouping_to_params[grouping_key].append(model_to_merge)

    varying_params = []
    for grouping_key, models in grouping_to_params.items():
        # Make sure the tasks in this grouping are unique.
        if len(models) == 1:
            logging.warning(
                f"Skipping merge for grouping key {grouping_key} since only 1 model was found."
            )
            continue
        assert len(set(p.task for p in models)) == 2

        (target_model,) = [
            m for m in models if m.fisher_run_uuid in target_fisher_run_uuids
        ]
        donor_models = [
            m for m in models if m.fisher_run_uuid in donor_fisher_run_uuids
        ]

        for donor_model in donor_models:
            varying_param = {
                "models_to_merge": [target_model, donor_model],
                "task": target_model.task,
                #
                "pretrained_model": grouping_key["pretrained_model"],
                "reg_strength": grouping_key["reg_strength"],
                "reg_type": grouping_key["reg_type"],
            }
            varying_params.append(varying_param)

    return varying_params


@experiment.experiment(
    uuid="a25c8aed42084d8cbc12458803d73f3d",
    group=BertMergingPrelimsGroup,
    params_cls=MergePairParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=finetune_bert_base.Glue_Regs,
        fisher_exp=bert_base_fisher_var_diag.RteMnliBestCkpt_Iso_0003_PhaseII,
        target_task="rte",
        donor_task="mnli",
    ),
    fixed_params={
        "fisher_type": "variational_diagonal",
        #
        "num_weightings": 19,
        "validation_examples": 2048,
        "sequence_length": 64,
        "batch_size": 2048,
    },
    key_fields={
        "models_to_merge",
        "num_weightings",
        "pretrained_model",
        "validation_examples",
    },
    bindings=[
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
class RteMnli_BestCkpt_Iso_0003_MergeSame_PhaseII(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )

    def create_preload_blob_uuids(self, params):
        return params.create_preload_blob_uuids()
