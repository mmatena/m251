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

from m251.models.bert import glue_classifier_execs as gc_exe
from m251.data.glue import glue

from ..group import BertMergingPrelimsGroup


TASKS = ("qqp", "sst2", "qnli", "mrpc", "rte")
REG_STRENGTHS = (1e-1, 1e-2, 3e-4, 0.0)


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
        sequence_length,
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
            scopes.ArgNameBindingSpec("sequence_length", self.sequence_length),
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
    uuid="4cc1e38d46f94786aa011cf4f536844d",
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
        "pretrained_model": "base",
        "batch_size": 8,
        "reg_type": "iso",
        # NOTE: See if we need 64 or 128 tokens for GLUE.
        "sequence_length": 64,
        # NOTE: Here None means use all training examples.
        "train_examples": None,
        "validation_examples": 4096,
        # NOTE: I should find a way to specific these cleaner.
        "examples_per_epoch": 25_000,
        "num_epochs": 200_000 // 25_000,
        "learning_rate": 1e-5,
    },
    key_fields={"pretrained_model", "task", "reg_strength", "reg_type"},
    bindings=[
        # For some reason, validation can cause the program to hang indefinitely.
        scopes.ArgNameBindingSpec("with_validation", False),
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
    ],
)
class FinetuneGlueIsoExperiment_Base(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )


############################################
############################################


@experiment.experiment(
    uuid="b413c236269047378eba6c41411ef815",
    group=BertMergingPrelimsGroup,
    params_cls=FinetuneIsoEvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    # NOTE: Create these partials from concise descriptors later.
    varying_params=functools.partial(
        create_varying_eval_params, train_exp=FinetuneGlueIsoExperiment_Base
    ),
    fixed_params={
        "batch_size": 16,
        "sequence_length": 64,
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
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
    ],
)
class FinetuneGlueIsoEval_Base(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )


###############################################################################
###############################################################################


@experiment.experiment(
    uuid="20213c6ee3c445acac63773d0b317cd0",
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
        "batch_size": 8,
        "reg_type": "iso",
        # NOTE: See if we need 64 or 128 tokens for GLUE.
        "sequence_length": 64,
        # NOTE: Here None means use all training examples.
        "train_examples": None,
        "validation_examples": 4096,
        # NOTE: I should find a way to specific these cleaner.
        "examples_per_epoch": 25_000,
        "num_epochs": 200_000 // 25_000,
        "learning_rate": 1e-5,
    },
    key_fields={"pretrained_model", "task", "reg_strength", "reg_type"},
    bindings=[
        # For some reason, validation can cause the program to hang indefinitely.
        scopes.ArgNameBindingSpec("with_validation", False),
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


############################################
############################################


@experiment.experiment(
    uuid="e66cf3f9c7db4a7196869112127169f2",
    group=BertMergingPrelimsGroup,
    params_cls=FinetuneIsoEvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    # NOTE: Create these partials from concise descriptors later.
    varying_params=functools.partial(
        create_varying_eval_params, train_exp=FinetuneGlueIsoExperiment_Large
    ),
    fixed_params={
        "batch_size": 8,
        "sequence_length": 64,
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
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
    ],
)
class FinetuneGlueIsoEval_Large(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )
