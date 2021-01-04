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

from m251.data.glue import glue
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

from m251.fisher.diagonal import diagonal_execs

from m251.models import model_execs
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import glue_metric_execs as metrics_exe

from ..group import BertMergingPrelimsGroup


BERT_BASE_DIAG_FISHER_UUID = "10b54ec1f7864c1791fdeb4facaf3681"


# @data_class.data_class()
# class FinetuneEwcParams(object):
#     def __init__(
#         self,
#         pretrained_model,
#         reg_type,
#         reg_strength,
#         train_examples,
#         batch_size,
#         task,
#         steps_per_epoch,
#         num_epochs,
#         sequence_length,
#         learning_rate,
#         fisher_matrix_uuid=None,
#         validation_examples=None,
#     ):
#         pass

#     def create_binding_specs(self):
#         if self.reg_type == "iso":
#             reg_bindings = [
#                 scopes.ArgNameBindingSpec(
#                     "regularizer", model_execs.regularize_body_l2_from_initial
#                 ),
#                 scopes.ArgNameBindingSpec("reg_strength", self.reg_strength),
#             ]
#         elif self.reg_type == "ewc":
#             reg_bindings = [
#                 scopes.ArgNameBindingSpec(
#                     "regularizer", diagonal_execs.diagonal_regularize_ewc_from_initial
#                 ),
#                 scopes.ArgNameBindingSpec("reg_strength", self.reg_strength),
#                 scopes.ArgNameBindingSpec("fisher_matrix_uuid", self.fisher_matrix_uuid),
#             ]
#         else:
#             raise ValueError(f"Invalid reg_type {self.reg_type}.")

#         return [
#             scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
#             scopes.ArgNameBindingSpec("train_num_examples", self.train_examples),
#             scopes.ArgNameBindingSpec(
#                 "validation_num_examples", self.validation_examples
#             ),
#             scopes.ArgNameBindingSpec("batch_size", self.batch_size),
#             scopes.ArgNameBindingSpec("tasks", [self.task]),
#             scopes.ArgNameBindingSpec("steps_per_epoch", self.steps_per_epoch),
#             scopes.ArgNameBindingSpec("epochs", self.num_epochs),
#             scopes.ArgNameBindingSpec("sequence_length", self.sequence_length),
#             scopes.ArgNameBindingSpec("learning_rate", self.learning_rate),
#         ] + reg_bindings

#     def create_preload_blob_uuids(self):
#         if self.fisher_matrix_uuid:
#             return (self.fisher_matrix_uuid,)
#         return None


# @experiment.experiment(
#     uuid="38bf99a99ac8487ea7bb9f5759e800b1",
#     group=BertMergingPrelimsGroup,
#     params_cls=FinetuneEwcParams,
#     executable_cls=fitting.training_run,
#     varying_params=[
#         {"task": "sst2", "reg_strength": 10.0 ** log_reg_str}
#         for log_reg_str in range(-5, 2 + 1)
#     ] + [{"task": "sst2", "reg_strength": 0.0}],
#     fixed_params={
#         "pretrained_model": "base",
#         #
#         "reg_type": "ewc",
#         "fisher_matrix_uuid": "10b54ec1f7864c1791fdeb4facaf3681",
#         #
#         "batch_size": 16,
#         "learning_rate": 1e-5,
#         "sequence_length": 64,
#         #
#         "steps_per_epoch": 512,
#         "num_epochs": 16,
#         #
#         "train_examples": None,
#         "validation_examples": 2048,
#     },
#     key_fields={"pretrained_model", "task", "reg_strength", "reg_type"},
#     bindings=[
#         # # For some reason, validation can cause the program to hang indefinitely.
#         # scopes.ArgNameBindingSpec("with_validation", False),
#         scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
#         scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
#         scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
#         scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
#         scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
#     ],
# )
# class GlueEwc_PhaseI(object):
#     """Prelminary experiment with goal of see what EWC regularization strengths to try."""
#     def create_run_instance_config(self, params):
#         return runs.RunInstanceConfig(
#             global_binding_specs=params.create_binding_specs()
#         )

#     def create_preload_blob_uuids(self, params):
#         return params.create_preload_blob_uuids()


###############################################################################
###############################################################################


GLUE_TASKS = ("qqp", "sst2", "qnli", "mrpc", "rte", "mnli", "cola", "stsb")
ISO_REG_STRENGTHS = (1e-1, 1e-2, 3e-4)
EWC_REG_STRENGTHS = (100.0, 10.0, 1.0, 0.03)


###############################################################################
###############################################################################


@data_class.data_class()
class FinetuneParams(object):
    def __init__(
        self,
        pretrained_model,
        reg_type,
        reg_strength,
        train_examples,
        batch_size,
        task,
        #
        num_task_epochs,
        num_ckpt_saves,
        #
        sequence_length,
        learning_rate,
        fisher_matrix_uuid=None,
        validation_examples=None,
    ):
        pass

    def create_binding_specs(self):
        if self.reg_type == "iso":
            reg_bindings = [
                scopes.ArgNameBindingSpec(
                    "regularizer", model_execs.regularize_body_l2_from_initial
                ),
                scopes.ArgNameBindingSpec("reg_strength", self.reg_strength),
            ]
        elif self.reg_type == "ewc":
            reg_bindings = [
                scopes.ArgNameBindingSpec(
                    "regularizer", diagonal_execs.diagonal_regularize_ewc_from_initial
                ),
                scopes.ArgNameBindingSpec("reg_strength", self.reg_strength),
                scopes.ArgNameBindingSpec(
                    "fisher_matrix_uuid", self.fisher_matrix_uuid
                ),
            ]
        else:
            raise ValueError(f"Invalid reg_type {self.reg_type}.")

        num_examples = self.num_task_epochs * NUM_GLUE_TRAIN_EXAMPLES[self.task]
        num_steps = int(round(num_examples / self.batch_size))

        keras_epochs = self.num_ckpt_saves
        keras_steps_per_epoch = int(round(num_steps / keras_epochs))

        return [
            scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
            scopes.ArgNameBindingSpec("train_num_examples", self.train_examples),
            scopes.ArgNameBindingSpec(
                "validation_num_examples", self.validation_examples
            ),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
            scopes.ArgNameBindingSpec("tasks", [self.task]),
            scopes.ArgNameBindingSpec("epochs", keras_epochs),
            scopes.ArgNameBindingSpec("steps_per_epoch", keras_steps_per_epoch),
            scopes.ArgNameBindingSpec("sequence_length", self.sequence_length),
            scopes.ArgNameBindingSpec("learning_rate", self.learning_rate),
        ] + reg_bindings

    def create_preload_blob_uuids(self):
        if self.fisher_matrix_uuid:
            return (self.fisher_matrix_uuid,)
        return None


############################################
############################################


@data_class.data_class()
class EvalParams(object):
    def __init__(
        self,
        #
        checkpoints_summary,
        #
        pretrained_model,
        task,
        reg_type,
        reg_strength,
        #
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

    def create_preload_blob_uuids(self):
        return self.checkpoints_summary.checkpoint_uuids


@experiment.with_experiment_storages()
def create_varying_eval_params(_, train_exp):
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
                "reg_strength": run_params.reg_strength,
                "reg_type": run_params.reg_type,
            }
        )

    return varying_params


###############################################################################
###############################################################################


@experiment.experiment(
    uuid="5b51eac957174646ac90beca59c0d6b9",
    group=BertMergingPrelimsGroup,
    params_cls=FinetuneParams,
    executable_cls=fitting.training_run,
    varying_params=(
        [
            {"reg_type": "iso", "task": task, "reg_strength": reg_str}
            for task in GLUE_TASKS
            for reg_str in ISO_REG_STRENGTHS
        ]
        + [
            {"reg_type": "ewc", "task": task, "reg_strength": reg_str}
            for task in GLUE_TASKS
            for reg_str in EWC_REG_STRENGTHS
        ]
        + [
            # Ignore the reg_type here, these are no regularization runs.
            {"reg_type": "iso", "task": task, "reg_strength": 0.0}
            for task in GLUE_TASKS
        ]
    ),
    fixed_params={
        "pretrained_model": "base",
        "fisher_matrix_uuid": BERT_BASE_DIAG_FISHER_UUID,
        #
        "batch_size": 16,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_task_epochs": 4,
        "num_ckpt_saves": 8,
        #
        "train_examples": None,
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
class Glue_Regs(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )

    def create_preload_blob_uuids(self, params):
        return params.create_preload_blob_uuids()


############################################
############################################


@experiment.experiment(
    uuid="289254cb58564891b47f494f9b70ff25",
    group=BertMergingPrelimsGroup,
    params_cls=EvalParams,
    executable_cls=eval_execs.evaluate_from_checkpoints_summary,
    varying_params=functools.partial(create_varying_eval_params, train_exp=Glue_Regs),
    fixed_params={
        "sequence_length": 64,
        "num_examples": 2048,
        "batch_size": 2048,
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
        #
        scopes.ArgNameBindingSpec("evaluate_model", eval_execs.robust_evaluate_model),
        scopes.ArgNameBindingSpec(
            "robust_evaluate_dataset", glue.glue_robust_evaluation_dataset
        ),
        scopes.ArgNameBindingSpec("metrics_for_tasks", metrics_exe.glue_robust_metrics),
        #
        scopes.ArgNameBindingSpec("cache_validation_batches_as_lists", True),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
    ],
)
class GlueEval_Regs(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )

    def create_preload_blob_uuids(self, params):
        return params.create_preload_blob_uuids()
