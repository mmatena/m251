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
from m251.fisher.execs import fisher_execs

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .finetune import GlueFinetune, GlueFinetune_LowRegStrength


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
    max_examples,
    alt_ckpt_fn=None,
):
    varying_params = []
    run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    for run_id in run_ids:
        run_params = train_exp.retrieve_run_params(run_id)

        checkpoints_summary = train_exp.retrieve_checkpoints_summary(run_id)

        if alt_ckpt_fn:
            ckpt_id = alt_ckpt_fn(run_params, checkpoints_summary)
            if not ckpt_id:
                continue
        else:
            ckpt_id = checkpoints_summary.checkpoint_uuids[-1]

        num_examples = min(NUM_GLUE_TRAIN_EXAMPLES[run_params.task], max_examples)

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


@experiment.experiment(
    uuid="f1f09aaa269c44069b3211d401ecb919",
    group=PaperExpGroup,
    params_cls=DirectFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=GlueFinetune,
        max_examples=4096,
    ),
    fixed_params={
        "batch_size": 1,
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
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 3),
    ],
)
class FisherComputation_LastCkpt(ExperimentAbc):
    pass


#######################################


def alt_ckpt_fn(params, ckpt_summary):
    if params.task == "qqp":
        # Looks like the qqp run might have gone sour between the second to
        # last and last checkpoints. So try the second to last checkpoint.
        return ckpt_summary.checkpoint_uuids[-2]
    return None


@experiment.experiment(
    uuid="a0e67992d6774801a443457023f40787",
    group=PaperExpGroup,
    params_cls=DirectFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=GlueFinetune,
        max_examples=4096,
        alt_ckpt_fn=alt_ckpt_fn,
    ),
    fixed_params={
        "batch_size": 1,
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
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 3),
    ],
)
class FisherComputation_AltCkpts(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="f4df570766c84f5e97c9d6a113c7a03e",
    group=PaperExpGroup,
    params_cls=DirectFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=GlueFinetune_LowRegStrength,
        max_examples=4096,
    ),
    fixed_params={
        "batch_size": 1,
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
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 3),
    ],
)
class FisherComputation_LowRegStrength(ExperimentAbc):
    pass
