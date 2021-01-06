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

from .finetune import GlueFinetune


class FisherParamsAbc(ParamsAbc):
    def create_common_bindings(self):
        return {
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


@data_class.data_class()
class VariationalFisherParams(FisherParamsAbc):
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
        beta,
        learning_rate,
    ):
        pass

    def get_train_steps(self):
        num_examples = self.num_examples
        if self.num_examples is None:
            num_examples = NUM_GLUE_TRAIN_EXAMPLES[self.task]
        examples_to_see = max(MIN_EXAMPLES_TO_SEE, num_examples)
        return examples_to_see // self.batch_size

    def create_bindings(self):
        return {
            "compiled_fisher_computer": vardiag_execs.variational_diag_fisher_computer,
            "variational_fisher_beta": self.variational_fisher_beta,
            "save_fisher_at_each_epoch": self.save_fisher_at_each_epoch,
            "epochs": 1,
            "steps_per_epoch": self.get_train_steps(),
            **self.create_common_bindings(),
        }


@data_class.data_class()
class DirectFisherParams(FisherParamsAbc):
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
    ):
        pass

    def create_bindings(self):
        return {
            "compiled_fisher_computer": diag_execs.diagonal_fisher_computer,
            **self.create_common_bindings(),
        }


###############################################################################


@experiment.with_experiment_storages()
def create_varying_params(
    exp,
    train_exp,
    task_to_example_counts,
):
    varying_params = []
    run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    for run_id in run_ids:
        run_params = train_exp.retrieve_run_params(run_id)

        checkpoints_summary = train_exp.retrieve_checkpoints_summary(run_id)
        ckpt_id = checkpoints_summary.checkpoint_uuids[-1]

        example_counts = task_to_example_counts[run_params.task]
        for num_examples in example_counts:
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
                    "num_examples": num_examples,
                }
            )

    return varying_params


###############################################################################


TASK_TO_EXAMPLE_COUNTS = {
    # None means all examples.
    "rte": [256, 1024, None],
    "mnli": [256, 1024, 4096, 32768, None],
}

MIN_EXAMPLES_TO_SEE = 8096


@experiment.experiment(
    uuid="41ed224169f4449e8bc11638e829cadc",
    group=PaperExpGroup,
    params_cls=VariationalFisherParams,
    executable_cls=fisher_execs.variational_fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=GlueFinetune,
        task_to_example_counts=TASK_TO_EXAMPLE_COUNTS,
    ),
    fixed_params={
        "batch_size": 8,
        "sequence_length": 64,
        #
        "beta": 1e-08,
        "learning_rate": 0.01,
    },
    key_fields={
        "trial_index",
        #
        "pretrained_model",
        "task",
        #
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
    ],
)
class VariationalFisherComputation(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="d35cc31aa6f34dd4b7c2dce43fa58028",
    group=PaperExpGroup,
    params_cls=DirectFisherParams,
    executable_cls=fisher_execs.variational_fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=GlueFinetune,
        task_to_example_counts=TASK_TO_EXAMPLE_COUNTS,
    ),
    fixed_params={
        "batch_size": 4,
        "sequence_length": 64,
    },
    key_fields={
        "trial_index",
        #
        "pretrained_model",
        "task",
        #
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
    ],
)
class DirectFisherComputation(ExperimentAbc):
    pass
