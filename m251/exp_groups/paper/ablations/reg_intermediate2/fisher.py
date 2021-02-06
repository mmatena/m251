"""TODO: Add title."""
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.training import optimizers

from m251.data.glue import glue
from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.diagonal import variational_diagonal_execs as vardiag_execs
from m251.fisher.execs import fisher_execs

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .finetune import Finetune_Rte, Finetune_Mnli


@data_class.data_class()
class DirectFisherParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        finetuned_run_uuid,
        finetuned_ckpt_uuid,
        #
        pretrained_model,
        task,
        #
        num_examples,
        #
        batch_size,
        sequence_length,
        #
        reg_type,
        reg_strength,
    ):
        pass

    def create_bindings(self):
        return {
            "compiled_fisher_computer": diag_execs.diagonal_fisher_computer,
            #
            "pretrained_model": self.pretrained_model,
            "tasks": [self.task],
            #
            "finetuned_run_uuid": self.finetuned_run_uuid,
            "finetuned_ckpt_uuid": self.finetuned_ckpt_uuid,
            #
            "num_examples": self.num_examples,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }

    def create_preload_blob_uuids(self):
        return (self.finetuned_ckpt_uuid,)


###############################################################################


@experiment.with_experiment_storages()
def create_varying_params(
    exp,
    train_exp,
    task_to_example_count,
):
    varying_params = []
    run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    for run_id in run_ids:
        run_params = train_exp.retrieve_run_params(run_id)

        checkpoints_summary = train_exp.retrieve_checkpoints_summary(run_id)
        ckpt_id = checkpoints_summary.checkpoint_uuids[-1]

        varying_params.append(
            {
                "trial_index": run_params.trial_index,
                #
                "task": run_params.task,
                "pretrained_model": run_params.pretrained_model,
                #
                "finetuned_run_uuid": run_id,
                "finetuned_ckpt_uuid": ckpt_id,
                #
                "num_examples": task_to_example_count[run_params.task],
                #
                "reg_type": run_params.reg_type,
                "reg_strength": run_params.reg_strength,
            }
        )

    return varying_params


###############################################################################


TASK_TO_EXAMPLE_COUNT = {
    # None means all examples.
    "rte": NUM_GLUE_TRAIN_EXAMPLES["rte"],
    "mnli": 4096,
}


@experiment.experiment(
    uuid="26490fb57063442b85585a99e376aa23",
    group=PaperExpGroup,
    params_cls=DirectFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_Rte,
        task_to_example_count=TASK_TO_EXAMPLE_COUNT,
    ),
    fixed_params={
        "batch_size": 4,
        "sequence_length": 64,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
)
class Fisher_Rte(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="84092b50c15a4bad8b2cafb1f43e6524",
    group=PaperExpGroup,
    params_cls=DirectFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_Mnli,
        task_to_example_count=TASK_TO_EXAMPLE_COUNT,
    ),
    fixed_params={
        "batch_size": 4,
        "sequence_length": 64,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
)
class Fisher_Mnli(ExperimentAbc):
    pass
