"""TODO: Add title."""
import collections
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.evaluation import eval_execs
from del8.executables.models import checkpoints
from del8.executables.training import fitting
from del8.executables.training import optimizers

from m251.data.domains import target_tasks

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.models import model_execs
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import glue_metric_execs as metrics_exe

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from m251.exp_groups.paper.results import utils as result_utils

from .eval2 import Eval_DAPT_LowResource_All
from .merge2 import Merge_MlmS2orc_ROBERTA


get_single_score = result_utils.get_single_score


###############################################################################
###############################################################################


@data_class.data_class()
class MergedEvalParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        target_ckpt_index,
        #
        models_to_merge,
        weighting,
        #
        sequence_length,
        batch_size,
        #
        pretrained_mlm_model,
        normalize_fishers=True,
    ):
        pass

    def create_bindings(self):
        return {
            "mergeable_model": diag_execs.diagonal_mergeable_model_from_checkpoint_or_pretrained,
            "model_merger": diag_execs.diagonal_model_merger,
            #
            "checkpoint_to_fisher_matrix_uuid": self.get_checkpoint_to_fisher_matrix_uuid(),
            "weightings": [self.weighting],
            #
            "normalize_fishers": self.normalize_fishers,
            #
            "checkpoints": [m.model_checkpoint_uuid for m in self.models_to_merge],
            "checkpoint_tasks": [m.task for m in self.models_to_merge],
            #
            "task": self.models_to_merge[0].task,
            "tasks": [m.task for m in self.models_to_merge],
            #
            "pretrained_model": self.pretrained_mlm_model,
            #
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            #
            "num_examples": None,
        }

    def get_checkpoint_to_fisher_matrix_uuid(self):
        key = (
            lambda m: m.model_checkpoint_uuid
            if m.model_checkpoint_uuid
            else self.pretrained_mlm_model
        )
        return {key(m): m.fisher_matrix_uuid for m in self.models_to_merge}

    def create_preload_blob_uuids(self):
        dikt = self.get_checkpoint_to_fisher_matrix_uuid()
        return tuple(
            (set(dikt.keys()) | set(dikt.values())) - {self.pretrained_mlm_model}
        )


###############################################################################


def create_varying_params_from_merge(
    exp,
    merge_exp,
    by_original_score=False,
    target_ckpt_indices=None,
    weighting_multiplier=None,
):
    with merge_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[merge_exp.uuid])

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=merge_exp.uuid)

    train_run_to_stuff = collections.defaultdict(list)
    for run_id in merge_run_ids:
        merge_run = exps_data.get_run_data(run_id)

        try:
            params = merge_run.get_single_item_by_class(merge_exp.params_cls)
        except AssertionError:
            continue

        if (
            target_ckpt_indices is not None
            and params.target_ckpt_index not in target_ckpt_indices
        ):
            continue

        reses = merge_run.get_items_by_class(merging_execs.MergingEvaluationResults)

        if by_original_score:
            (res,) = [r for r in reses if r.weighting[0] == 1.0]
        else:
            res = max(reses, key=lambda r: get_single_score(r.results))

        train_run_uuid = params.models_to_merge[0].train_run_uuid

        train_run_to_stuff[train_run_uuid].append(
            {
                "params": params,
                "best": res,
            }
        )

    best_per_train_run = [
        max(stuff, key=lambda s: get_single_score(s["best"].results))
        for stuff in train_run_to_stuff.values()
    ]

    varying_params = []
    for stuff in best_per_train_run:
        params, best = stuff["params"], stuff["best"]
        # print(params.pretrained_mlm_model)
        weighting = best.weighting
        if weighting_multiplier is not None:
            weighting = [weighting[0] * weighting_multiplier]
            weighting.append(1 - weighting[0])
        varying_params.append(
            {
                "trial_index": params.trial_index,
                "target_ckpt_index": params.target_ckpt_index,
                "pretrained_mlm_model": params.pretrained_mlm_model,
                "models_to_merge": params.models_to_merge,
                "weighting": weighting,
            }
        )

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="877e92857bcd4ae18b622b0b4dcedb17",
    group=PaperExpGroup,
    params_cls=MergedEvalParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params_from_merge,
        merge_exp=Merge_MlmS2orc_ROBERTA,
    ),
    fixed_params={
        "sequence_length": 256,
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
        scopes.ArgNameBindingSpec("split", "test"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", target_tasks.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class EvalTest_Merged_MlmS2orc_ROBERTA_Best(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="123b1cf3e96f4cc882d235ac2dd8efb1",
    group=PaperExpGroup,
    params_cls=MergedEvalParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params_from_merge,
        merge_exp=Merge_MlmS2orc_ROBERTA,
        target_ckpt_indices={5},
    ),
    fixed_params={
        "sequence_length": 256,
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
        scopes.ArgNameBindingSpec("split", "test"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", target_tasks.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class EvalTest_Merged_MlmS2orc_ROBERTA_Ckpt5(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="c702e2708ffe4c059dacaa5698a185cd",
    group=PaperExpGroup,
    params_cls=MergedEvalParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params_from_merge,
        merge_exp=Merge_MlmS2orc_ROBERTA,
        target_ckpt_indices={9},
    ),
    fixed_params={
        "sequence_length": 256,
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
        scopes.ArgNameBindingSpec("split", "test"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", target_tasks.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class EvalTest_Merged_MlmS2orc_ROBERTA_Ckpt9(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="b8ba7c7544804ee49b31b30b010a112a",
    group=PaperExpGroup,
    params_cls=MergedEvalParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params_from_merge,
        merge_exp=Merge_MlmS2orc_ROBERTA,
        target_ckpt_indices={9},
        weighting_multiplier=0.5,
    ),
    fixed_params={
        "sequence_length": 256,
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
        scopes.ArgNameBindingSpec("split", "test"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", target_tasks.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class EvalTest_Merged_MlmS2orc_ROBERTA_Ckpt9_HalfWeight(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="bec82093066843318b55dd13ac9ba6bc",
    group=PaperExpGroup,
    params_cls=MergedEvalParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params_from_merge,
        merge_exp=Merge_MlmS2orc_ROBERTA,
        by_original_score=True,
    ),
    fixed_params={
        "sequence_length": 256,
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
        scopes.ArgNameBindingSpec("split", "test"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", target_tasks.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class EvalTest_Original_MlmS2orc_ROBERTA_Best(ExperimentAbc):
    pass


###############################################################################
###############################################################################


@data_class.data_class()
class EvalParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        ckpt_uuid,
        #
        pretrained_model,
        task,
        #
        batch_size,
        sequence_length,
    ):
        pass

    def create_bindings(self):
        return {
            "checkpoints_summary": checkpoints.CheckpointsSummary(
                checkpoint_uuids=[self.ckpt_uuid]
            ),
            "pretrained_model": self.pretrained_model,
            "tasks": [self.task],
            "num_examples": None,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
        }

    def create_preload_blob_uuids(self):
        return (self.ckpt_uuid,)


###############################################################################


def create_varying_params_from_eval(
    exp,
    eval_exp,
):
    with eval_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[eval_exp.uuid])

    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=eval_exp.uuid)

    varying_params = []
    for run_id in run_ids:
        run = exps_data.get_run_data(run_id)

        params = run.get_single_item_by_class(eval_exp.params_cls)
        reses = run.get_items_by_class(eval_execs.CheckpointEvaluationResults)

        res = max(reses, key=lambda r: get_single_score(r.results))
        print(params.pretrained_model)
        varying_params.append(
            {
                "trial_index": params.trial_index,
                "ckpt_uuid": res.checkpoint_blob_uuid,
                "pretrained_model": params.pretrained_model,
                "task": params.task,
            }
        )

    return varying_params


###############################################################################


@experiment.experiment(
    uuid="73fa024af71b469bb19aae84613542c3",
    group=PaperExpGroup,
    params_cls=EvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    varying_params=functools.partial(
        create_varying_params_from_eval,
        eval_exp=Eval_DAPT_LowResource_All,
    ),
    fixed_params={
        "sequence_length": 256,
        "batch_size": 128,
    },
    key_fields={
        "trial_index",
        "task",
        "pretrained_model",
    },
    bindings=[
        scopes.ArgNameBindingSpec("split", "test"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", target_tasks.robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class EvalTest_Original_DAPT_Best(ExperimentAbc):
    pass
