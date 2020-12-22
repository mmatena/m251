"""TODO: Add title."""
from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs

from del8.executables.data import tfds as tfds_execs
from del8.executables.models import checkpoints as ckpt_exec
from del8.executables.training import fitting
from del8.executables.training import optimizers

from m251.models.bert import glue_classifier_execs as gc_exe
from m251.data.glue import glue

from ..group import BertMergingPrelimsGroup


TASKS = ("qqp", "sst2", "qnli")
REG_STRENGTHS = (1.0, 1e-1, 1e-2, 3e-4, 0.0)


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
        examples_per_epoch,
        num_epochs,
        sequence_length,
        validation_examples,
        learning_rate,
    ):
        pass

    def create_binding_specs(self):
        steps_per_epoch = round(self.examples_per_epoch / self.batch_size)

        if self.reg_type == "iso":
            reg_bindings = [
                scopes.ArgNameBindingSpec(
                    "regularizer", gc_exe.regularize_body_l2_from_initial
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
            scopes.ArgNameBindingSpec("steps_per_epoch", steps_per_epoch),
            scopes.ArgNameBindingSpec("epochs", self.num_epochs),
            scopes.ArgNameBindingSpec("sequence_length", self.sequence_length),
            scopes.ArgNameBindingSpec("learning_rate", self.learning_rate),
        ] + reg_bindings


@experiment.experiment(
    uuid="87e969507fc341df992badcc14c7adf1",
    group=BertMergingPrelimsGroup,
    params_cls=FinetuneIsoParams,
    executable_cls=fitting.training_run,
    # NOTE: Create these partials from concise descriptors later.
    varying_params=[
        {"task": task, "reg_strength": reg_str}
        for task in TASKS
        for reg_str in REG_STRENGTHS
    ],
    fixed_params={
        "pretrained_model": "large",
        # NOTE: Here None means use all training examples.
        "batch_size": 8,
        # NOTE: See if we need 64 or 128 tokens for GLUE.
        "reg_type": "iso",
        "sequence_length": 64,
        "train_examples": None,
        "validation_examples": 4096,
        # NOTE: I should find a way to specific these cleaner.
        "examples_per_epoch": 25_000,
        "num_epochs": 200_000 // 25_000,
        "learning_rate": 3e-5,
    },
    key_fields={"pretrained_model", "task", "reg_strength", "reg_type"},
    bindings=[
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
    ],
)
class FinetuneGlueIsoExperiment_Large(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )
