"""TODO: Add title."""
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.evaluation import eval_execs
from del8.executables.models import checkpoints as ckpt_exec
from del8.executables.training import fitting
from del8.executables.training import optimizers

from m251.data.image import image_classification
from m251.models import model_execs
from m251.models.simclr import simclr
from m251.models.simclr import simclr_classifier_execs as sc_exe

from ..simclr_group import SimclrMergingPrelimsGroup


# NOTE: Skipping caltech101 as I am having trouble downloading it.
TASKS = ("cifar100", "dtd", "oxford_iiit_pet")
REG_STRENGTHS = (0.0, 1e-6, 3e-5, 1e-3)


@data_class.data_class()
class FinetuneIsoParams(object):
    def __init__(
        self,
        pretrained_model,
        reg_type,
        reg_strength,
        train_examples,
        batch_size,
        task,
        train_steps,
        steps_per_epoch,
        image_size,
        validation_examples,
        learning_rate,
    ):
        pass

    def create_binding_specs(self):
        num_epochs = round(self.train_steps / self.steps_per_epoch)

        if self.reg_type == "iso":
            reg_bindings = [
                scopes.ArgNameBindingSpec(
                    "regularizer", model_execs.regularize_body_l2_from_initial
                ),
                scopes.ArgNameBindingSpec("reg_strength", self.reg_strength),
            ]
        else:
            raise ValueError(f"Invalid reg_type {self.reg_type}.")

        return [
            scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
            scopes.ArgNameBindingSpec("train_num_examples", self.train_examples),
            scopes.ArgNameBindingSpec(
                "validation_num_examples", self.validation_examples
            ),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
            scopes.ArgNameBindingSpec("tasks", [self.task]),
            scopes.ArgNameBindingSpec("steps_per_epoch", self.steps_per_epoch),
            scopes.ArgNameBindingSpec("epochs", num_epochs),
            scopes.ArgNameBindingSpec("image_size", self.image_size),
            scopes.ArgNameBindingSpec("learning_rate", self.learning_rate),
        ] + reg_bindings


############################################
############################################


@data_class.data_class()
class FinetuneIsoEvalParams(object):
    def __init__(
        self,
        checkpoints_summary,
        pretrained_model,
        task,
        reg_type,
        reg_strength,
        num_examples,
        batch_size,
        image_size,
    ):
        pass

    def create_binding_specs(self):

        return [
            scopes.ArgNameBindingSpec("checkpoints_summary", self.checkpoints_summary),
            scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
            scopes.ArgNameBindingSpec("tasks", [self.task]),
            scopes.ArgNameBindingSpec("reg_type", self.reg_type),
            scopes.ArgNameBindingSpec("reg_strength", self.reg_strength),
            scopes.ArgNameBindingSpec("num_examples", self.num_examples),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
            scopes.ArgNameBindingSpec("image_size", self.image_size),
        ]


def create_varying_eval_params(eval_exp, train_exp):
    run_uuids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    varying_params = []
    for run_uuid in run_uuids:
        run_params = train_exp.retrieve_run_params(run_uuid)
        ckpt_summary = train_exp.retrieve_checkpoints_summary(run_uuid)
        varying_params.append(
            {
                "checkpoints_summary": ckpt_summary,
                "pretrained_model": run_params.pretrained_model,
                "task": run_params.task,
                # NOTE: Not needed for eval, but add so we can have the same
                # run key between the training and evaluation runs.
                "reg_strength": run_params.reg_strength,
                "reg_type": run_params.reg_type,
            }
        )
    return varying_params


###############################################################################
###############################################################################


@experiment.experiment(
    uuid="f22e2f050b23450e901d8836db17098d",
    group=SimclrMergingPrelimsGroup,
    params_cls=FinetuneIsoParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {"task": task, "reg_strength": reg_str}
        for task in TASKS
        for reg_str in REG_STRENGTHS
    ],
    fixed_params={
        "pretrained_model": "r50_1x",
        "batch_size": 32,
        "reg_type": "iso",
        "image_size": simclr.IMAGE_SIZE,
        "train_examples": None,
        "validation_examples": 4096,
        "train_steps": 80_000,
        "steps_per_epoch": 10_000,
        "learning_rate": 1e-3,
    },
    key_fields={"pretrained_model", "task", "reg_strength", "reg_type"},
    bindings=[
        # For some reason, validation can cause the program to hang indefinitely.
        scopes.ArgNameBindingSpec("with_validation", False),
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        scopes.ArgNameBindingSpec("compiled_model", sc_exe.simclr_finetuning_model),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
    ],
)
class FinetuneSimclrIso_r50_1x(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )


############################################
############################################


@experiment.experiment(
    uuid="e6639409b748455187c011c87c7ae5c6",
    group=SimclrMergingPrelimsGroup,
    params_cls=FinetuneIsoEvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    varying_params=functools.partial(
        create_varying_eval_params, train_exp=FinetuneSimclrIso_r50_1x
    ),
    fixed_params={
        "batch_size": 32,
        "image_size": simclr.IMAGE_SIZE,
        "num_examples": 4096,
    },
    key_fields={
        # Same as from the training experiment.
        "pretrained_model",
        "task",
        "reg_strength",
        "reg_type",
    },
    bindings=[
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("compiled_model", sc_exe.simclr_finetuning_model),
    ],
)
class FinetuneSimclrIsoEval_r50_1x(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )
