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

from . import flop_util


@data_class.data_class()
class FinetuneParams(ParamsAbc):
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
        num_task_epochs,
        num_ckpt_saves,
    ):
        pass

    def create_bindings(self):
        num_examples = self.num_task_epochs * NUM_GLUE_TRAIN_EXAMPLES[self.task]
        num_steps = int(round(num_examples / self.batch_size))

        keras_epochs = self.num_ckpt_saves
        keras_steps_per_epoch = int(round(num_steps / keras_epochs))

        return {
            "pretrained_model": self.pretrained_model,
            "tasks": [self.task],
            #
            "reg_strength": self.reg_strength,
            #
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            #
            "epochs": keras_epochs,
            "steps_per_epoch": keras_steps_per_epoch,
        }


###############################################################################


@experiment.experiment(
    uuid="0227e6bbdbff43939bd7abd2a8cdcce9",
    group=PaperExpGroup,
    params_cls=FinetuneParams,
    # executable_cls=flop_util.compute_eval_flops,
    executable_cls=flop_util.compute_train_flops,
    varying_params=[{"task": "rte", "trial_index": 0, "reg_strength": 0.0}],
    fixed_params={
        "pretrained_model": "bert-base-uncased",
        #
        "reg_type": "iso",
        #
        "batch_size": 16,
        "learning_rate": 1e-5,
        "sequence_length": 64,
        #
        "num_task_epochs": 10,
        "num_ckpt_saves": 4,
    },
    key_fields={
        "trial_index",
        "task",
        "reg_strength",
    },
    bindings=[
        scopes.ArgNameBindingSpec("with_validation", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("callbacks", ckpt_exec.checkpoint_saver_callback),
        scopes.ArgNameBindingSpec("compiled_model", gc_exe.bert_finetuning_model),
        #
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
        scopes.ArgNameBindingSpec("metrics", None),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
    ],
)
class Finetune_Rte(ExperimentAbc):
    pass
