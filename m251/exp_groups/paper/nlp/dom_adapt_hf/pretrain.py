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

from m251.data.domains import s2orc

from m251.models import model_execs
from m251.models.bert import glue_classifier_execs as gc_exe
from m251.models.bert import glue_metric_execs as metrics_exe
from m251.models.bert import bert as bert_common
from m251.models.bert import roberta_mlm_execs as mlm_execs

from m251.exp_groups.paper.paper_group import ExperimentAbc
from m251.exp_groups.paper.paper_group import ParamsAbc

from m251.exp_groups.paper.paper_group import PaperExpGroup


S2ORC_TASKS = ["cs", "bio_med"]

S2ORC_TASK_TO_DAPT_NAME = {
    "bio_med": "allenai/biomed_roberta_base",
    "cs": "allenai/cs_roberta_base",
}


@data_class.data_class()
class PretrainParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        #
        pretrained_model,
        task,
        #
        reg_type,
        reg_strength,
        #
        learning_rate,
        sequence_length,
        batch_size,
        #
        num_examples,
        num_ckpt_saves,
    ):
        pass

    def create_bindings(self):
        train_steps = self.num_examples // self.batch_size
        keras_epochs = self.num_ckpt_saves
        keras_steps_per_epoch = train_steps // self.num_ckpt_saves

        return {
            "pretrained_model": self.pretrained_model,
            "task": self.task,
            "tasks": [self.task],
            #
            "regularizer": model_execs.regularize_body_l2_from_initial,
            "reg_strength": self.reg_strength,
            #
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            #
            "epochs": keras_epochs,
            "steps_per_epoch": keras_steps_per_epoch,
        }


@experiment.experiment(
    uuid="85b76abce61544fbaf322689189df3e4",
    group=PaperExpGroup,
    params_cls=PretrainParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {
            "task": task,
        }
        for task in ["cs"]
    ],
    fixed_params={
        "trial_index": 0,
        #
        "pretrained_model": "roberta-base",
        #
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 32,
        "learning_rate": 1e-5,
        "sequence_length": 256,
        #
        "num_ckpt_saves": 1,
        "num_examples": 32768,
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
        scopes.ArgNameBindingSpec("with_validation", False),
        #
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
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        #
        scopes.ArgNameBindingSpec("loss", mlm_execs.roberta_mlm_loss),
        scopes.ArgNameBindingSpec("metrics", None),
    ],
)
class Pretrain_32768(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="78e844c762924893ac357a352fabec7d",
    group=PaperExpGroup,
    params_cls=PretrainParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {
            "task": task,
            "num_examples": num_examples,
        }
        for task in ["cs"]
        for num_examples in [262144, 2097152]
    ],
    fixed_params={
        "trial_index": 0,
        #
        "pretrained_model": "roberta-base",
        #
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 32,
        "learning_rate": 1e-5,
        "sequence_length": 256,
        #
        "num_ckpt_saves": 1,
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
        scopes.ArgNameBindingSpec("with_validation", False),
        #
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
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        #
        scopes.ArgNameBindingSpec("loss", mlm_execs.roberta_mlm_loss),
        scopes.ArgNameBindingSpec("metrics", None),
    ],
)
class Pretrain_More(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="b687bb3e29484ef9a943c4946cb9d0dd",
    group=PaperExpGroup,
    params_cls=PretrainParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {
            "task": task,
            "pretrained_model": S2ORC_TASK_TO_DAPT_NAME[task],
        }
        for task in ["cs"]
    ],
    fixed_params={
        "trial_index": 0,
        #
        "reg_type": "iso",
        "reg_strength": 3e-4,
        #
        "batch_size": 32,
        "learning_rate": 1e-5,
        "sequence_length": 256,
        #
        "num_ckpt_saves": 1,
        #
        "num_examples": 32768,
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
        scopes.ArgNameBindingSpec("with_validation", False),
        #
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
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        #
        scopes.ArgNameBindingSpec("loss", mlm_execs.roberta_mlm_loss),
        scopes.ArgNameBindingSpec("metrics", None),
        #
        scopes.ArgNameBindingSpec("max_paragraphs_per_doc", 1),
    ],
)
class Pretrain_FromDapt_32768(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="4ffd3e962eea4a068cd5b969f719de77",
    group=PaperExpGroup,
    params_cls=PretrainParams,
    executable_cls=fitting.training_run,
    varying_params=[
        {
            "task": task,
        }
        for task in ["cs"]
    ],
    fixed_params={
        "trial_index": 0,
        #
        "pretrained_model": "roberta-base",
        #
        "reg_type": "iso",
        "reg_strength": 0.0,
        #
        "batch_size": 32,
        "learning_rate": 1e-5,
        "sequence_length": 256,
        #
        "num_ckpt_saves": 1,
        "num_examples": 32768,
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
        scopes.ArgNameBindingSpec("with_validation", False),
        #
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
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        #
        scopes.ArgNameBindingSpec("loss", mlm_execs.roberta_mlm_loss),
        scopes.ArgNameBindingSpec("metrics", None),
        #
        scopes.ArgNameBindingSpec("max_paragraphs_per_doc", 1),
    ],
)
class Pretrain_32768_NoReg(ExperimentAbc):
    pass
