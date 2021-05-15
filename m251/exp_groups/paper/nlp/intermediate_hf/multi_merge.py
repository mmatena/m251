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

from .fisher import (
    FisherComputation_BertBase_HighResource,
    FisherComputation_BertBase_LowResource_LastCkpt,
    FisherComputation_BertBaseFromMnliCkpt_LastCkpt,
    FisherComputation_BertBase_Squad2,
)


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
        chunk_index,
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
                self.num_weightings, seed=self.chunk_index
            ),
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


def create_varying_params(
    exp,
    target_tasks,
    donor_tasks,
    target_fisher_exp,
    donor_fisher_exps,
    num_chunks,
):
    with exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[e.uuid for e in donor_fisher_exps]
            + [target_fisher_exp.uuid]
        )

    target_run_params, target_fishers = _get_infos_for_task(
        exps_data, target_fisher_exp
    )

    donor_run_params, donor_fishers = _get_infos_for_task(exps_data, donor_fisher_exps)

    fishers = target_fishers.copy()
    fishers.update(donor_fishers)

    varying_params = []
    for p in target_run_params:
        trial_index = p.trial_index
        if p.task not in target_tasks:
            continue
        mtm = _to_mtm(p, fishers)
        for p1, p2 in itertools.combinations(donor_run_params, 2):

            if p1.task not in donor_tasks or p2.task not in donor_tasks:
                continue
            elif p.uuid == p1.uuid and p.uuid == p2.uuid:
                continue
            elif p1.task == p2.task:
                continue

            mtm1 = _to_mtm(p1, fishers)
            mtm2 = _to_mtm(p2, fishers)
            print(trial_index, p1.task, p2.task)

            for chunk_index in range(num_chunks):
                varying_params.append(
                    {
                        "trial_index": trial_index,
                        "models_to_merge": [mtm, mtm1, mtm2],
                        "chunk_index": chunk_index,
                    }
                )
    random.shuffle(varying_params)
    return varying_params


###############################################################################


@experiment.experiment(
    uuid="c5ac525b604d4b9c8acd129716c2f033",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        target_tasks={"rte"},
        donor_tasks={"mnli", "qnli"},
        target_fisher_exp=FisherComputation_BertBase_LowResource_LastCkpt,
        donor_fisher_exps=[FisherComputation_BertBase_HighResource],
        num_chunks=50,
    ),
    fixed_params={
        "pretrained_model": "bert-base-uncased",
        #
        "num_weightings": 50,
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
        "chunk_index",
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
class Merge_BertBase_Rte_MnliQnli(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="9705957901bd48af8569808b2d00444a",
    group=PaperExpGroup,
    params_cls=MergeParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_params,
        target_tasks={"rte"},
        donor_tasks={"mnli", "rte"},
        target_fisher_exp=FisherComputation_BertBase_LowResource_LastCkpt,
        donor_fisher_exps=[
            FisherComputation_BertBase_HighResource,
            FisherComputation_BertBase_LowResource_LastCkpt,
        ],
        num_chunks=4,
    ),
    fixed_params={
        "pretrained_model": "bert-base-uncased",
        #
        "num_weightings": 150,
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
        "chunk_index",
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
class Merge_BertBase_Rte_MnliRte(ExperimentAbc):
    pass
