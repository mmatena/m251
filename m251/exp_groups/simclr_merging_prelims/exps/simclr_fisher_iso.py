"""TODO: Add title."""
import functools

from absl import logging

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs

from m251.fisher.diagonal import diagonal_execs
from m251.fisher.execs import fisher_execs

from m251.data.image import image_classification
from m251.models import model_execs
from m251.models.simclr import simclr
from m251.models.simclr import simclr_classifier_execs as sc_exe

from . import simclr_finetune_iso
from ..simclr_group import SimclrMergingPrelimsGroup


@data_class.data_class()
class FisherIsoParams(object):
    def __init__(
        self,
        finetuned_exp_uuid,
        finetuned_run_uuid,
        finetuned_ckpt_uuid,
        finetuned_ckpt_index,
        pretrained_model,
        task,
        num_examples,
        batch_size,
        fisher_type,
        diagonal_y_samples,
        image_size,
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
            scopes.ArgNameBindingSpec("image_size", self.image_size),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
        ] + fisher_bindings


def create_varying_params(exp, ft_exp, checkpoint_index):
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

        ckpt_summary = ft_exp.retrieve_checkpoints_summary(run_uuid)
        varying_params.append(
            {
                "finetuned_exp_uuid": ft_exp.uuid,
                "finetuned_run_uuid": run_uuid,
                "finetuned_ckpt_uuid": ckpt_summary.checkpoint_uuids[checkpoint_index],
                "finetuned_ckpt_index": checkpoint_index,
                "pretrained_model": run_params.pretrained_model,
                "task": run_params.task,
            }
        )
    return varying_params


###############################################################################
###############################################################################


# Model r50_1x, checkpoint from 20k steps in.
@experiment.experiment(
    uuid="f9f5b732653146ee922f11bc2d7b5f70",
    group=SimclrMergingPrelimsGroup,
    params_cls=FisherIsoParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        ft_exp=simclr_finetune_iso.FinetuneSimclrIso_r50_1x,
        # 20k steps in.
        checkpoint_index=1,
    ),
    fixed_params={
        "batch_size": 1,
        "num_examples": 4096,
        "fisher_type": "diagonal",
        # Compute y expectation exactly.
        "diagonal_y_samples": None,
        "image_size": simclr.IMAGE_SIZE,
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
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        scopes.ArgNameBindingSpec("initializer", sc_exe.simclr_initializer),
        scopes.ArgNameBindingSpec("builder", sc_exe.simclr_builder),
        scopes.ArgNameBindingSpec(
            "metrics", model_execs.multitask_classification_metrics
        ),
    ],
)
class SimclrFisherIso__r50_1x__ckpt_20k(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )
