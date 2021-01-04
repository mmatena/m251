"""TODO: Add title."""
import collections
import itertools
import functools

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
from m251.data.processing.constants import GLUE_POSITIVE_TRANSFER_TASKS

from m251.fisher.diagonal import diagonal_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.models import model_execs
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import glue_metric_execs as metrics_exe

from .diag_merge_glue_iso import ModelToMerge
from ..group import BertMergingPrelimsGroup

from . import finetune_bert_base
from . import fisher_bert_base


BERT_BASE_DIAG_FISHER_UUID = "10b54ec1f7864c1791fdeb4facaf3681"


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
        fisher_params,
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
        if self.fisher_type == "diagonal":
            fisher_bindings = [
                scopes.ArgNameBindingSpec(
                    "mergeable_model",
                    diagonal_execs.diagonal_mergeable_model_from_checkpoint,
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
def create_varying_params_all_mnli_to_rte(
    exp, mnli_fisher_exp, rte_fisher_exp, train_exp
):
    train_run_uuids = train_exp.retrieve_run_uuids(RunState.FINISHED)
    mnli_fisher_run_uuids = mnli_fisher_exp.retrieve_run_uuids(RunState.FINISHED)
    rte_fisher_run_uuids = rte_fisher_exp.retrieve_run_uuids(RunState.FINISHED)

    train_run_params = {
        rid: train_exp.retrieve_run_params(rid) for rid in train_run_uuids
    }

    mnli_fisher_run_params = [
        mnli_fisher_exp.retrieve_run_params(rid) for rid in mnli_fisher_run_uuids
    ]
    rte_fisher_run_params = [
        rte_fisher_exp.retrieve_run_params(rid) for rid in rte_fisher_run_uuids
    ]

    grouping_to_params = collections.defaultdict(list)
    for fi_rid, fi_rp in zip(
        mnli_fisher_run_uuids + rte_fisher_run_uuids,
        mnli_fisher_run_params + rte_fisher_run_params,
    ):
        assert fi_rp.ft_exp_uuid == train_exp.uuid

        if fi_rp.task not in ["mnli", "rte"]:
            continue
        # NOTE: I forgot to add this before launching the experiment.
        elif fi_rp.task == "mnli" and fi_rid not in mnli_fisher_run_uuids:
            continue

        tr_rp = train_run_params[fi_rp.ft_run_uuid]

        if fi_rid in mnli_fisher_run_uuids:
            fisher_exp = mnli_fisher_exp
        else:
            fisher_exp = rte_fisher_exp

        grouping_key = {}
        grouping_key.update(fisher_exp.create_run_key_values(fi_rp))
        grouping_key.update(train_exp.create_run_key_values(tr_rp))
        del grouping_key["finetuned_run_uuid"]
        del grouping_key["finetuned_ckpt_uuid"]
        del grouping_key["task"]
        grouping_key = hashabledict(grouping_key)

        saved_fisher_matrix = fisher_exp.retrieve_single_item_by_class(
            fisher_execs.SavedFisherMatrix, fi_rid
        )

        model_to_merge = ModelToMerge(
            task=tr_rp.task,
            train_run_uuid=fi_rp.ft_run_uuid,
            fisher_run_uuid=fi_rid,
            model_checkpoint_uuid=fi_rp.finetuned_ckpt_uuid,
            fisher_matrix_uuid=saved_fisher_matrix.blob_uuid,
        )

        grouping_to_params[grouping_key].append(model_to_merge)

    varying_params = []
    for grouping_key, models in grouping_to_params.items():

        base_param = {
            "pretrained_model": grouping_key["pretrained_model"],
            "reg_type": grouping_key["reg_type"],
            "reg_strength": grouping_key["reg_strength"],
            "fisher_type": grouping_key["fisher_type"],
        }

        if grouping_key["fisher_type"] == "diagonal":
            base_param["fisher_params"] = {
                "y_samples": grouping_key["diagonal_y_samples"],
            }
        else:
            raise ValueError(f"Invalid fisher_type {grouping_key['fisher_type']}.")

        (rte_model,) = [mtm for mtm in models if mtm.task == "rte"]
        mnli_models = [mtm for mtm in models if mtm.task == "mnli"]
        for model in mnli_models:
            varying_param = base_param.copy()
            varying_param.update(
                {
                    "models_to_merge": [rte_model, model],
                    "task": "rte",
                }
            )
            varying_params.append(varying_param)

    return varying_params


@experiment.experiment(
    uuid="40925b89452e46129f14d58dd4f0385c",
    group=BertMergingPrelimsGroup,
    params_cls=MergePairParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params_all_mnli_to_rte,
        train_exp=finetune_bert_base.Glue_Regs,
        rte_fisher_exp=fisher_bert_base.GlueRegs_Fisher_BestCkpt,
        mnli_fisher_exp=fisher_bert_base.GlueRegs_Fisher_Mnli_AllCkpt,
    ),
    fixed_params={
        "num_weightings": 19,
        "validation_examples": 2048,
        "sequence_length": 64,
        "batch_size": 2048,
    },
    key_fields={
        "models_to_merge",
        "num_weightings",
        "pretrained_model",
        "fisher_type",
        "fisher_params",
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
class MergeBestRteWithEachMnli_GlueRegs(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )

    def create_preload_blob_uuids(self, params):
        return params.create_preload_blob_uuids()
