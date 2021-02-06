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

from m251.data.domains import target_tasks
from m251.data.domains import s2orc

from m251.fisher.diagonal import diagonal_execs as diag_execs
from m251.fisher.execs import fisher_execs

from m251.models import model_execs
from m251.models.bert import bert as bert_common
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import roberta_mlm_execs as mlm_execs

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup

from .pretrain import PretrainCs, Pretrain_BioMed


@data_class.data_class()
class MlmFisherParams(ParamsAbc):
    def __init__(
        self,
        #
        pretrained_model,
        task,
        num_examples,
        y_samples,
        #
        batch_size,
        sequence_length,
        #
        trial_index,
        #
        checkpoint,
        pretrained_run_uuid,
        #
        pretrained_examples,
        pretrained_reg_strength,
    ):
        pass

    def create_bindings(self):
        if self.num_examples is None:
            raise ValueError(
                "You need to set num_examples to full dataset size instead of setting it to None."
            )
        return {
            "compiled_fisher_computer": diag_execs.diagonal_fisher_computer,
            "y_samples": self.y_samples,
            #
            "pretrained_model": self.pretrained_model,
            "task": self.task,
            "tasks": [self.task],
            "num_examples": self.num_examples,
            #
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            #
            "checkpoint": self.checkpoint,
        }


@experiment.with_experiment_storages()
def create_varying_params(
    exp,
    train_exp,
    fisher_examples,
    all_ckpts=False,
    pretrained_examples=None,
    pretrained_reg_strength=None,
):
    varying_params = []
    run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    for run_id in run_ids:
        run_params = train_exp.retrieve_run_params(run_id)

        if (
            pretrained_examples is not None
            and run_params.num_examples != pretrained_examples
        ):
            continue
        if (
            pretrained_reg_strength is not None
            and run_params.reg_strength != pretrained_reg_strength
        ):
            continue

        checkpoints_summary = train_exp.retrieve_checkpoints_summary(run_id)

        if all_ckpts:
            ckpts = checkpoints_summary.checkpoint_uuids
        else:
            ckpts = checkpoints_summary.checkpoint_uuids[-1:]

        for checkpoint_uuid in ckpts:
            varying_params.append(
                {
                    "trial_index": 0,
                    #
                    "task": run_params.task,
                    "pretrained_model": run_params.pretrained_model,
                    #
                    "checkpoint": checkpoint_uuid,
                    "pretrained_run_uuid": run_id,
                    #
                    "num_examples": fisher_examples,
                    #
                    "pretrained_examples": run_params.num_examples,
                    "pretrained_reg_strength": run_params.reg_strength,
                }
            )

    return varying_params


@experiment.experiment(
    uuid="740e68e6450d4285a56ed5d0ba924137",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=PretrainCs,
        fisher_examples=16384,
    ),
    fixed_params={
        "sequence_length": 256,
        #
        "batch_size": 2,
        "y_samples": 1,
    },
    key_fields={
        "trial_index",
        #
        "checkpoint",
        "task",
        #
        "num_examples",
        "y_samples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", s2orc.mlm_dataset),
        #
        scopes.ArgNameBindingSpec("compiled_model", mlm_execs.roberta_mlm_model),
        #
        scopes.ArgNameBindingSpec("tokenizer", bert_common.bert_tokenizer),
        scopes.ArgNameBindingSpec("initializer", mlm_execs.roberta_initializer),
        scopes.ArgNameBindingSpec("loader", mlm_execs.roberta_loader),
        scopes.ArgNameBindingSpec("builder", mlm_execs.roberta_builder),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
        #
        scopes.ArgNameBindingSpec("max_paragraphs_per_doc", 1),
    ],
)
class Fisher_PretrainCs_16384(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="f1a345dbcd664f9e967ad31779a2085f",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Pretrain_BioMed,
        fisher_examples=16384,
    ),
    fixed_params={
        "sequence_length": 256,
        #
        "batch_size": 2,
        "y_samples": 1,
    },
    key_fields={
        "trial_index",
        #
        "checkpoint",
        "task",
        #
        "num_examples",
        "y_samples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", s2orc.mlm_dataset),
        #
        scopes.ArgNameBindingSpec("compiled_model", mlm_execs.roberta_mlm_model),
        #
        scopes.ArgNameBindingSpec("tokenizer", bert_common.bert_tokenizer),
        scopes.ArgNameBindingSpec("initializer", mlm_execs.roberta_initializer),
        scopes.ArgNameBindingSpec("loader", mlm_execs.roberta_loader),
        scopes.ArgNameBindingSpec("builder", mlm_execs.roberta_builder),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", False),
        #
        scopes.ArgNameBindingSpec("max_paragraphs_per_doc", 1),
    ],
)
class Fisher_PretrainBioMed_16384(ExperimentAbc):
    pass
