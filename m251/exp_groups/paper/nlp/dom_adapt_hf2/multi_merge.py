"""TODO: Add title."""
import collections
import functools
import itertools
import random

import tensorflow as tf
import tensorflow_probability as tfp

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

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.diagonal import dummy_execs
from m251.fisher.diagonal import variational_diagonal_execs as vardiag_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.models.bert import bert as bert_common
from m251.models.bert import glue_metric_execs as metrics_exe
from m251.models.bert import roberta_mlm_execs as mlm_execs

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import create_pairwise_weightings
from m251.exp_groups.paper.paper_group import ModelToMerge

from m251.exp_groups.paper.paper_group import PaperExpGroup

from ..dom_adapt_hf.fisher2 import (
    FisherComputation_ROBERTA_TargetTasks_LastCkpt_AllVars,
    FisherComputation_ROBERTA_TargetTasks_AllCkpts,
)
from .fisher import Fisher_PretrainCs_16384, Fisher_PretrainBioMed_16384
from .target_fisher import (
    Fisher_Cs_32768_1e6,
    Fisher_DAPT_CsFt_AllCkpts,
    Fisher_DAPT_BioMedFt_AllCkpts,
)

from ..dom_adapt_hf.fisher3 import Fisher_PretrainFromDapt32768

from ..intermediate_hf.fisher import FisherComputation_RobertaBase_Mnli


def create_threeway_weightings(num_weightings, seed=735492):
    tf.random.set_seed(seed)
    dist = tfp.distributions.Dirichlet(tf.ones([3]))
    return dist.sample(num_weightings, seed=seed).numpy().tolist()


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
            "model_merger": diag_execs.diagonal_model_merger,
            #
            "checkpoint_to_fisher_matrix_uuid": self.get_checkpoint_to_fisher_matrix_uuid(),
            "weightings": create_threeway_weightings(
                self.num_weightings, seed=self.trial_index
            ),
            #
            "checkpoints": [m.model_checkpoint_uuid for m in self.models_to_merge],
            "checkpoint_tasks": [m.task for m in self.models_to_merge],
            "additional_model_bindings": [
                m.additional_model_bindings for m in self.models_to_merge
            ],
            #
            "task": self.models_to_merge[0].task,
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
        }

    def get_checkpoint_to_fisher_matrix_uuid(self):
        return {
            m.model_checkpoint_uuid: m.fisher_matrix_uuid for m in self.models_to_merge
        }


###############################################################################


def _to_mtm(run_params, fishers):
    # Supports both fine-tuned and downloaded models.
    train_run_uuid = getattr(
        run_params,
        "finetuned_run_uuid",
        getattr(run_params, "pretrained_run_uuid", None),
    )
    model_checkpoint_uuid = getattr(
        run_params,
        "finetuned_ckpt_uuid",
        getattr(
            run_params, "checkpoint", getattr(run_params, "pretrained_model", None)
        ),
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


def create_varying_params(
    exp,
    target_fisher_exp,
    donor_fisher_exps,
    tasks,
    donor_tasks=("cs", "mnli"),
    pretrained_examples=None,
    pretrained_reg_strength=None,
):
    with exp.get_storage() as storage:
        target_exps_data = storage.retrieve_storage_data(
            experiment_uuid=[target_fisher_exp.uuid]
        )
        donor_exps_data = storage.retrieve_storage_data(
            experiment_uuid=[e.uuid for e in donor_fisher_exps]
        )

    target_run_params, target_fishers = _get_infos_for_task(
        target_exps_data, target_fisher_exp
    )
    donor_run_params, donor_fishers = _get_infos_for_task(
        donor_exps_data, donor_fisher_exps
    )
    fishers = target_fishers.copy()
    fishers.update(donor_fishers)

    varying_params = []
    for p in target_run_params:
        if p.task not in tasks:
            continue
        for p1, p2 in itertools.combinations(donor_run_params, 2):
            if hasattr(p2, "pretrained_examples"):
                p1, p2 = p2, p1
            if p1.task not in donor_tasks or p2.task not in donor_tasks:
                continue
            elif p.uuid == p1.uuid and p.uuid == p2.uuid:
                continue
            elif p1.task == p2.task:
                continue

            if (
                pretrained_examples is not None
                and p1.pretrained_examples != pretrained_examples
            ):
                continue
            elif (
                pretrained_reg_strength is not None
                and p1.pretrained_reg_strength != pretrained_reg_strength
            ):
                continue

            mtm = _to_mtm(p, fishers)
            mtm1 = _to_mtm(p1, fishers)
            mtm2 = _to_mtm(p2, fishers)

            # Hack so we can get the MLM weights in a glue classifier.
            mtm1 = mtm1.copy(
                additional_model_bindings=[
                    # ("model", mlm_execs.roberta_mlm_model),
                    ("initializer", mlm_execs.roberta_initializer),
                    ("loader", mlm_execs.roberta_loader),
                    ("builder", mlm_execs.roberta_builder),
                    ("tokenizer", bert_common.bert_tokenizer),
                ]
            )
            # mtm2 = mtm2.copy(
            #     additional_model_bindings=[
            #         # ("model", mlm_execs.roberta_mlm_model),
            #         ("initializer", mlm_execs.roberta_initializer),
            #         ("loader", mlm_execs.roberta_loader),
            #         ("builder", mlm_execs.roberta_builder),
            #         ("tokenizer", bert_common.bert_tokenizer),
            #     ]
            # )

            varying_params.append(
                {
                    "trial_index": p.trial_index,
                    "models_to_merge": [mtm, mtm1, mtm2],
                    "pretrained_model": p1.pretrained_model,
                }
            )
    return varying_params


###############################################################################


CS_TASKS = ["acl_arc", "sci_erc"]
BIO_MED_TASKS = ["chemprot"]


@experiment.experiment(
    uuid="8cd6c4594e454b10a908c997a7f8565c",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        target_fisher_exp=FisherComputation_ROBERTA_TargetTasks_AllCkpts,
        donor_fisher_exps=[Fisher_PretrainCs_16384, FisherComputation_RobertaBase_Mnli],
        tasks=CS_TASKS,
        donor_tasks=("cs", "mnli"),
        pretrained_examples=1048576,
        pretrained_reg_strength=0.0,
    ),
    fixed_params={
        "num_weightings": 251,
        #
        "validation_examples": 2048,
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
        #
        scopes.ArgNameBindingSpec("min_fisher", 1e-20),
    ],
)
class Merge_ROBERTA_AllCkpts_TestSet_PretrainCs(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="e411515139d14053bbd19573132b3e1a",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        target_fisher_exp=FisherComputation_ROBERTA_TargetTasks_AllCkpts,
        donor_fisher_exps=[
            Fisher_PretrainBioMed_16384,
            FisherComputation_RobertaBase_Mnli,
        ],
        tasks=BIO_MED_TASKS,
        donor_tasks=("bio_med", "mnli"),
        pretrained_examples=1048576,
        pretrained_reg_strength=0.0,
    ),
    fixed_params={
        "num_weightings": 251,
        #
        "validation_examples": 2048,
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
        #
        scopes.ArgNameBindingSpec("min_fisher", 1e-20),
    ],
)
class Merge_ROBERTA_AllCkpts_TestSet_PretrainBioMed(ExperimentAbc):
    pass