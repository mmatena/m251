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

from .pretrain import (
    Pretrain_32768,
    Pretrain_More,
    Pretrain_FromDapt_32768,
    Pretrain_32768_NoReg,
)


S2ORC_TASKS = ["cs", "bio_med"]

S2ORC_TASK_TO_DAPT_NAME = {
    "bio_med": "allenai/biomed_roberta_base",
    "cs": "allenai/cs_roberta_base",
}


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
        pretrained_examples=None,
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
def create_varying_params(exp, train_exp, fisher_examples):
    varying_params = []
    run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    for run_id in run_ids:
        run_params = train_exp.retrieve_run_params(run_id)

        checkpoints_summary = train_exp.retrieve_checkpoints_summary(run_id)

        checkpoint_uuid = checkpoints_summary.checkpoint_uuids[-1]
        for num_examples in fisher_examples:
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
                    "num_examples": num_examples,
                    #
                    "pretrained_examples": run_params.num_examples,
                }
            )

    return varying_params


@experiment.experiment(
    uuid="072238630325429388472f3238ebcc89",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Pretrain_32768,
        fisher_examples=[16384, 8 * 16384],
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
        "pretrained_model",
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
    ],
)
class Fisher_PretrainMore(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="0ca78e3a71694d158e19261625e31691",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Pretrain_More,
        fisher_examples=[16384, 8 * 16384],
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
    ],
)
class Fisher_PretrainMore_REAL(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="c3556f83761c40728ac56897c54035ac",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Pretrain_FromDapt_32768,
        fisher_examples=[16384],
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
class Fisher_PretrainFromDapt32768(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="fed1ec8e452c418aa244d5a8ce50ee25",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params, train_exp=Pretrain_32768_NoReg, fisher_examples=[16384]
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
class Fisher_Pretrain32768NoReg(ExperimentAbc):
    pass
