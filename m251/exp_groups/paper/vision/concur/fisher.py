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

from m251.data.image import image_classification
from m251.models import model_execs
from m251.models.simclr import simclr
from m251.models.simclr import simclr_classifier_execs as sc_exe

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.execs import fisher_execs

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .finetune import Finetune_Subsets
from . import defs


MAX_EXAMPLES = 4096


@data_class.data_class()
class FisherParams(ParamsAbc):
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
        skip_examples,
        #
        batch_size,
        image_size,
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
            "dataset_skip": self.skip_examples,
            "image_size": self.image_size,
            "batch_size": self.batch_size,
        }

    def create_preload_blob_uuids(self):
        return (self.finetuned_ckpt_uuid,)


@experiment.with_experiment_storages()
def create_varying_params_last_ckpt(
    exp,
    train_exp,
    max_examples,
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
                "num_examples": min(max_examples, run_params.train_examples),
                "skip_examples": run_params.train_skip_examples,
            }
        )

    return varying_params


@experiment.experiment(
    uuid="20d1fdb4e787416aaacf70aa5518b0ab",
    group=PaperExpGroup,
    params_cls=FisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params_last_ckpt,
        train_exp=Finetune_Subsets,
        max_examples=MAX_EXAMPLES,
    ),
    fixed_params={
        "batch_size": 4,
        "image_size": simclr.IMAGE_SIZE,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 4),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("initializer", sc_exe.simclr_initializer),
        scopes.ArgNameBindingSpec("loader", sc_exe.simclr_loader),
        scopes.ArgNameBindingSpec("builder", sc_exe.simclr_builder),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
)
class FisherComputation_Subsets_LastCkpt(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="dd2440972f054061bd6c29af9993700e",
    group=PaperExpGroup,
    params_cls=FisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params_last_ckpt,
        train_exp=Finetune_Subsets,
        max_examples=MAX_EXAMPLES,
    ),
    fixed_params={
        "batch_size": 4,
        "image_size": simclr.IMAGE_SIZE,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 4),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("initializer", sc_exe.simclr_initializer),
        scopes.ArgNameBindingSpec("loader", sc_exe.simclr_loader),
        scopes.ArgNameBindingSpec("builder", sc_exe.simclr_builder),
        #
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        #
        scopes.ArgNameBindingSpec("all_variables_mergeable", True),
    ],
)
class FisherComputation_Subsets_LastCkpt_AllVars(ExperimentAbc):
    pass
