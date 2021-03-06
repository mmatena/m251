"""TODO: Add title."""
import functools

from absl import logging

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs

from m251.data.glue import glue
from m251.fisher.diagonal import diagonal_execs
from m251.fisher.execs import fisher_execs

from m251.storage_util import eval_results

from . import finetune_glue_iso
from ..group import BertMergingPrelimsGroup


@data_class.data_class()
class FisherIsoParams(object):
    def __init__(
        self,
        finetuned_exp_uuid,
        finetuned_run_uuid,
        finetuned_ckpt_uuid,
        pretrained_model,
        task,
        num_examples,
        batch_size,
        fisher_type,
        diagonal_y_samples,
        sequence_length,
        fisher_class_chunk_size=None,
    ):
        pass

    @property
    def ft_exp_uuid(self):
        return self.finetuned_exp_uuid

    @property
    def ft_run_uuid(self):
        return self.finetuned_run_uuid

    def create_binding_specs(self):
        if self.fisher_type == "diagonal":
            fisher_bindings = [
                scopes.ArgNameBindingSpec(
                    "compiled_fisher_computer", diagonal_execs.diagonal_fisher_computer
                ),
                scopes.ArgNameBindingSpec("fisher_type", self.fisher_type),
                # NOTE: When I have more scopes, I should probably try to bind
                # this more strongly to where it is used.
                scopes.ArgNameBindingSpec("y_samples", self.diagonal_y_samples),
            ]
            if self.fisher_class_chunk_size is not None:
                fisher_bindings.append(
                    scopes.ArgNameBindingSpec(
                        "fisher_class_chunk_size", self.fisher_class_chunk_size
                    )
                )
        else:
            raise ValueError(f"Invalid fisher_type {self.fisher_type}.")

        return [
            scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
            scopes.ArgNameBindingSpec("tasks", [self.task]),
            #
            scopes.ArgNameBindingSpec("finetuned_exp_uuid", self.finetuned_exp_uuid),
            scopes.ArgNameBindingSpec("finetuned_run_uuid", self.finetuned_run_uuid),
            scopes.ArgNameBindingSpec("finetuned_ckpt_uuid", self.finetuned_ckpt_uuid),
            #
            scopes.ArgNameBindingSpec("num_examples", self.num_examples),
            scopes.ArgNameBindingSpec("sequence_length", self.sequence_length),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
        ] + fisher_bindings


def create_varying_params(exp, ft_exp, eval_exp=None, checkpoint="last"):
    # Purpose of the context manager is so that we don't keep reconnecting to GCP.
    ft_storage = ft_exp.get_storage()
    eval_storage = eval_exp.get_storage() if eval_exp else ft_storage
    with ft_storage, eval_storage:
        return _create_varying_params(
            exp, ft_exp, eval_exp=eval_exp, checkpoint=checkpoint
        )


def _create_varying_params(exp, ft_exp, eval_exp=None, checkpoint="last"):
    run_uuids = ft_exp.retrieve_run_uuids(RunState.FINISHED)

    varying_params = []
    run_keys = []
    for run_uuid in run_uuids:
        run_params = ft_exp.retrieve_run_params(run_uuid)
        # TODO: Make sure that no run keys are duplicated. Throw error now,
        # consider how to handle it in the future.
        run_key = ft_exp.retrieve_run_key(run_uuid)

        if any(runs.RunKey.has_same_values(rk, run_key) for rk in run_keys):
            # NOTE: I throw error now. In the future, I should think of what to do.
            # I'm thinking a first pass would be to manually remove the "bad" existing
            # one from the database. Also provide an option just to do it for each
            # duplicate. Also if I get this error, make sure it isn't being caused by
            # some misconfiguration.
            raise ValueError(
                f"Found duplicated run key {run_key.key_values} in the database. "
                "See comment in code above this raise statement for thoughts on what to do."
            )
        run_keys.append(run_key)

        if checkpoint == "last":
            ckpt_summary = ft_exp.retrieve_checkpoints_summary(run_uuid)
            ckpt = ckpt_summary.checkpoint_uuids[-1]
        elif checkpoint == "best":
            eval_run_uuid = eval_results.get_eval_run_uuid_for_train_run(
                ft_exp, eval_exp, run_uuid
            )
            best = eval_results.get_best_result_for_run(
                eval_exp, eval_run_uuid, run_params.task
            )
            ckpt = best.checkpoint_blob_uuid
        else:
            raise ValueError(checkpoint)

        varying_params.append(
            {
                "finetuned_exp_uuid": ft_exp.uuid,
                "finetuned_run_uuid": run_uuid,
                "finetuned_ckpt_uuid": ckpt,
                "pretrained_model": run_params.pretrained_model,
                "task": run_params.task,
            }
        )
    return varying_params


# Base, last checkpoint.
@experiment.experiment(
    uuid="e5235a8f536b4ad799896b7dfbdda8e4",
    group=BertMergingPrelimsGroup,
    params_cls=FisherIsoParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params, ft_exp=finetune_glue_iso.FinetuneGlueIsoExperiment_Base
    ),
    fixed_params={
        "batch_size": 4,
        "num_examples": 4096,
        "fisher_type": "diagonal",
        # Compute y expectation exactly.
        "diagonal_y_samples": None,
        "sequence_length": 64,
    },
    key_fields={
        "finetuned_run_uuid",
        "finetuned_ckpt_uuid",
        "fisher_type",
        "diagonal_y_samples",
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
    ],
)
class GlueFisherIsoExperiment_LastCkpt_Base(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )


# Large, last checkpoint.
@experiment.experiment(
    uuid="0f0932ad3fae428298df3226a7e650f8",
    group=BertMergingPrelimsGroup,
    params_cls=FisherIsoParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params, ft_exp=finetune_glue_iso.FinetuneGlueIsoExperiment_Large
    ),
    fixed_params={
        "batch_size": 2,
        "num_examples": 4096,
        "fisher_type": "diagonal",
        # Compute y expectation exactly.
        "diagonal_y_samples": None,
        "sequence_length": 64,
    },
    key_fields={
        "finetuned_run_uuid",
        "finetuned_ckpt_uuid",
        "fisher_type",
        "diagonal_y_samples",
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
    ],
)
class GlueFisherIsoExperiment_LastCkpt_Large(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )


# Large, best checkpoint.
@experiment.experiment(
    uuid="4b30fc47fdee406cad586743d1ce6e96",
    group=BertMergingPrelimsGroup,
    params_cls=FisherIsoParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        ft_exp=finetune_glue_iso.FinetuneGlueIsoExperiment_Large,
        eval_exp=finetune_glue_iso.FinetuneGlueIsoEval_Large,
        checkpoint="best",
    ),
    fixed_params={
        "batch_size": 2,
        "num_examples": 4096,
        "fisher_type": "diagonal",
        # Compute y expectation exactly.
        "diagonal_y_samples": None,
        "sequence_length": 64,
    },
    key_fields={
        "finetuned_run_uuid",
        "finetuned_ckpt_uuid",
        "fisher_type",
        "diagonal_y_samples",
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
    ],
)
class GlueFisherIsoExperiment_BestCkpt_Large(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )


# Large, best checkpoint.
@experiment.experiment(
    uuid="43f7f0cbafbc4b6fa5bda7a1536fbd69",
    group=BertMergingPrelimsGroup,
    params_cls=FisherIsoParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        ft_exp=finetune_glue_iso.FinetuneGlueIsoExperiment_RobertaLarge,
        eval_exp=finetune_glue_iso.FinetuneGlueIsoRobustEval_RobertaLarge,
        checkpoint="best",
    ),
    fixed_params={
        # "batch_size": 2,
        # STS-B.
        "batch_size": 1,
        "fisher_class_chunk_size": 4,
        "num_examples": 4096,
        "fisher_type": "diagonal",
        # Compute y expectation exactly.
        "diagonal_y_samples": None,
        "sequence_length": 64,
    },
    key_fields={
        "finetuned_run_uuid",
        "finetuned_ckpt_uuid",
        "fisher_type",
        "diagonal_y_samples",
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
    ],
)
class GlueFisherIsoExperiment_BestCkpt_RobertaLarge(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )
