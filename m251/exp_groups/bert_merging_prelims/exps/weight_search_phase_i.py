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
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

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


@data_class.data_class()
class WeightSearchParams(object):
    def __init__(
        self,
        #
        models_to_merge,
        #
        search_steps,
        search_num_examples,
        final_evaluate_num_examples,
        #
        fisher_type,
        fisher_params,
        #
        sequence_length,
        batch_size,
        #
        pretrained_model,
        task=None,
        reg_strength=None,
        reg_type=None,
        #
        search_num_inits=None,
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
                    "merge_weighting_searcher",
                    diagonal_execs.diagonal_model_merge_weighting_search,
                ),
                scopes.ArgNameBindingSpec(
                    "model_merger", diagonal_execs.diagonal_model_merger
                ),
            ]
        else:
            raise ValueError(f"Invalid fisher_type {self.fisher_type}.")

        return [
            scopes.ArgNameBindingSpec(
                "merge_weighting_search_steps", self.search_steps
            ),
            scopes.ArgNameBindingSpec(
                "merge_weighting_num_inits", self.search_num_inits
            ),
            scopes.ArgNameBindingSpec("search_num_examples", self.search_num_examples),
            scopes.ArgNameBindingSpec(
                "final_evaluate_num_examples", self.final_evaluate_num_examples
            ),
            #
            scopes.ArgNameBindingSpec(
                "checkpoint_to_fisher_matrix_uuid",
                self.get_checkpoint_to_fisher_matrix_uuid(),
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
            scopes.ArgNameBindingSpec("tasks", [self.models_to_merge[0].task]),
            #
            scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
            #
            scopes.ArgNameBindingSpec("sequence_length", self.sequence_length),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
        ] + fisher_bindings

    def create_preload_blob_uuids(self):
        dikt = self.get_checkpoint_to_fisher_matrix_uuid()
        return tuple(set(dikt.keys()) | set(dikt.values()))


@experiment.with_experiment_storages()
def create_varying_weight_search_phase_i_params(
    exp,
    fisher_exp,
    train_exp,
    target_task,
    donor_task_groups,
    reg_types,
    reg_strengths,
):
    train_run_uuids = train_exp.retrieve_run_uuids(RunState.FINISHED)
    fisher_run_uuids = fisher_exp.retrieve_run_uuids(RunState.FINISHED)

    train_run_params = {
        rid: train_exp.retrieve_run_params(rid) for rid in train_run_uuids
    }
    fisher_run_params = [
        fisher_exp.retrieve_run_params(rid) for rid in fisher_run_uuids
    ]

    grouping_to_params = collections.defaultdict(list)
    for fi_rid, fi_rp in zip(fisher_run_uuids, fisher_run_params):
        assert fi_rp.ft_exp_uuid == train_exp.uuid

        tr_rp = train_run_params[fi_rp.ft_run_uuid]

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
        # Make sure the tasks in this grouping are unique.
        assert len(set(p.task for p in models)) == len(models)

        if (
            grouping_key["reg_type"] not in reg_types
            or grouping_key["reg_strength"] not in reg_strengths
        ):
            continue

        base_params = {
            "pretrained_model": grouping_key["pretrained_model"],
            "reg_type": grouping_key["reg_type"],
            "reg_strength": grouping_key["reg_strength"],
            "fisher_type": grouping_key["fisher_type"],
        }

        if grouping_key["fisher_type"] == "diagonal":
            base_params["fisher_params"] = {
                "y_samples": grouping_key["diagonal_y_samples"],
            }
        else:
            raise ValueError(f"Invalid fisher_type {grouping_key['fisher_type']}.")

        (target_model,) = [mtm for mtm in models if mtm.task == target_task]

        for donor_tasks in donor_task_groups:
            donor_models = [mtm for mtm in models if mtm.task in donor_tasks]
            varying_param = base_params.copy()
            varying_param["models_to_merge"] = [target_model] + donor_models
            varying_params.append(varying_param)

    return varying_params


###############################################################################
###############################################################################


def _common_exp_args(search_steps=50, search_num_inits=3, **overrides):
    ret = {
        "group": BertMergingPrelimsGroup,
        "params_cls": WeightSearchParams,
        "executable_cls": merging_execs.merge_weighting_search_from_checkpoints,
        # 'varying_params': functools.partial(
        #     create_varying_weight_search_phase_i_params,
        #     train_exp=finetune_bert_base.Glue_Regs,
        #     fisher_exp=fisher_bert_base.GlueRegs_Fisher_BestCkpt,
        #     target_task="rte",
        #     max_models=3,
        # ),
        "fixed_params": {
            "search_num_inits": search_num_inits,
            "search_steps": search_steps,
            "search_num_examples": 1024,
            "final_evaluate_num_examples": 2048,
            "sequence_length": 64,
            "batch_size": 1024,
        },
        "key_fields": {
            "pretrained_model",
            "task",
            "reg_strength",
            "reg_type",
            #
            "fisher_type",
            #
            "search_steps",
            "search_num_inits",
            #
            "models_to_merge",
        },
        "bindings": [
            scopes.ArgNameBindingSpec("split", "validation"),
            scopes.ArgNameBindingSpec("shuffle", False),
            scopes.ArgNameBindingSpec("repeat", False),
            #
            scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
            scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
            #
            scopes.ArgNameBindingSpec(
                "evaluate_model", eval_execs.robust_evaluate_model
            ),
            scopes.ArgNameBindingSpec(
                "robust_evaluate_dataset", glue.glue_robust_evaluation_dataset
            ),
            scopes.ArgNameBindingSpec(
                "metrics_for_tasks", metrics_exe.glue_robust_metrics
            ),
            #
            scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        ],
    }
    ret.update(overrides)
    return ret


class _BaseExp(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )

    def create_preload_blob_uuids(self, params):
        return params.create_preload_blob_uuids()


_COMMON_PARAMS_PARTIAL_KWARGS = {
    "train_exp": finetune_bert_base.Glue_Regs,
    "fisher_exp": fisher_bert_base.GlueRegs_Fisher_BestCkpt,
    "reg_types": ("iso"),
    "reg_strengths": (0.0003, 0.01),
    # 'reg_strengths': (0.0, 0.0003, 0.01),
}


###############################################################################
###############################################################################


@experiment.experiment(
    uuid="1114c9479ac94e67ba37c6495a10f4f4",
    varying_params=functools.partial(
        create_varying_weight_search_phase_i_params,
        target_task="rte",
        donor_task_groups=[
            ("mnli"),
            ("qnli"),
            ("stsb"),
        ],
        **_COMMON_PARAMS_PARTIAL_KWARGS,
    ),
    **_common_exp_args(
        search_steps=100,
    ),
)
class MergeWeightSearch_GP_RTE_2_Tasks_PhaseI(_BaseExp):
    pass


@experiment.experiment(
    uuid="889db2b8721e40cab5b7d895d9d9ad20",
    varying_params=functools.partial(
        create_varying_weight_search_phase_i_params,
        target_task="rte",
        donor_task_groups=[
            ("mnli", "qnli"),
            ("mnli", "stsb"),
            ("qnli", "stsb"),
        ],
        **_COMMON_PARAMS_PARTIAL_KWARGS,
    ),
    **_common_exp_args(
        search_steps=100,
    ),
)
class MergeWeightSearch_GP_RTE_3_Tasks_PhaseI(_BaseExp):
    pass


@experiment.experiment(
    uuid="48ebcf39409c4f1eb4f35059bb869a04",
    varying_params=functools.partial(
        create_varying_weight_search_phase_i_params,
        target_task="mrpc",
        donor_task_groups=[
            ("qqp", "mnli", "qnli"),
        ],
        **_COMMON_PARAMS_PARTIAL_KWARGS,
    ),
    **_common_exp_args(
        # search_steps=350,
        search_steps=150,
    ),
)
class MergeWeightSearch_GP_MRPC_HighResourceInformedTasks_PhaseI(_BaseExp):
    pass


@experiment.experiment(
    uuid="bea4bbc3a31b47b0869de76093f04999",
    varying_params=functools.partial(
        create_varying_weight_search_phase_i_params,
        target_task="stsb",
        donor_task_groups=[
            ("qqp", "qnli"),
        ],
        **_COMMON_PARAMS_PARTIAL_KWARGS,
    ),
    **_common_exp_args(
        # search_steps=350,
        search_steps=100,
    ),
)
class MergeWeightSearch_GP_STSB_HighResourceInformedTasks_PhaseI(_BaseExp):
    pass
