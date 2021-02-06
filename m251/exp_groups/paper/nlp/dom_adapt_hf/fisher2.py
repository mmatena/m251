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

from .finetune2 import Finetune_DAPT_LowResource
from .finetune2 import Finetune_ROBERTA_LowResource
from .finetune2 import Finetune_DAPT_LowResource_HeadOnly


TRAIN_EXAMPLES = target_tasks.TRAIN_EXAMPLES
TASK_TO_DAPT_NAME = target_tasks.TASK_TO_DAPT_NAME

LOW_RESOURCE_TASKS = ["chemprot", "acl_arc", "sci_erc"]


@data_class.data_class()
class TargetTaskFisherParams(ParamsAbc):
    def __init__(
        self,
        #
        trial_index,
        checkpoint_index,
        #
        finetuned_run_uuid,
        finetuned_ckpt_uuid,
        #
        pretrained_model,
        task,
        num_examples,
        y_samples,
        #
        batch_size,
        sequence_length,
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
            "finetuned_run_uuid": self.finetuned_run_uuid,
            "finetuned_ckpt_uuid": self.finetuned_ckpt_uuid,
            #
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
        }

    def create_preload_blob_uuids(self):
        return (self.finetuned_ckpt_uuid,)


@experiment.with_experiment_storages()
def create_varying_params(exp, train_exp, task_to_example_count, min_ckpt_index=0):
    varying_params = []
    run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)

    for run_id in run_ids:
        run_params = train_exp.retrieve_run_params(run_id)

        checkpoints_summary = train_exp.retrieve_checkpoints_summary(run_id)

        checkpoint_uuids = checkpoints_summary.checkpoint_uuids[min_ckpt_index:]
        for ckpt_index, ckpt_id in enumerate(checkpoint_uuids):

            print(run_params.pretrained_model)

            varying_params.append(
                {
                    "trial_index": run_params.trial_index,
                    "checkpoint_index": min_ckpt_index + ckpt_index,
                    #
                    "task": run_params.task,
                    "pretrained_model": run_params.pretrained_model,
                    #
                    "finetuned_run_uuid": run_id,
                    "finetuned_ckpt_uuid": ckpt_id,
                    #
                    "num_examples": task_to_example_count[run_params.task],
                }
            )

    return varying_params


@experiment.experiment(
    uuid="29753f37bf2e4fa5a65c5df5f01d2516",
    group=PaperExpGroup,
    params_cls=TargetTaskFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_ROBERTA_LowResource,
        task_to_example_count=TRAIN_EXAMPLES,
    ),
    fixed_params={
        "batch_size": 2,
        "sequence_length": 256,
        #
        "y_samples": None,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 4),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class FisherComputation_ROBERTA_TargetTasks(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="f9bc2c36e23641829f8892c97bde4f68",
    group=PaperExpGroup,
    params_cls=TargetTaskFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_ROBERTA_LowResource,
        task_to_example_count=TRAIN_EXAMPLES,
        min_ckpt_index=9,
    ),
    fixed_params={
        "batch_size": 2,
        "sequence_length": 256,
        #
        "y_samples": None,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 4),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
        #
        scopes.ArgNameBindingSpec("all_variables_mergeable", True),
    ],
)
class FisherComputation_ROBERTA_TargetTasks_LastCkpt_AllVars(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="ac02591a39bb4dd4bc07ce2c0492a103",
    group=PaperExpGroup,
    params_cls=TargetTaskFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_ROBERTA_LowResource,
        task_to_example_count=TRAIN_EXAMPLES,
        min_ckpt_index=0,
    ),
    fixed_params={
        "batch_size": 2,
        "sequence_length": 256,
        #
        "y_samples": None,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 4),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
    ],
)
class FisherComputation_ROBERTA_TargetTasks_AllCkpts(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="142627f5881f44a588738fec0b532d04",
    group=PaperExpGroup,
    params_cls=TargetTaskFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=functools.partial(
        create_varying_params,
        train_exp=Finetune_DAPT_LowResource_HeadOnly,
        task_to_example_count=TRAIN_EXAMPLES,
        min_ckpt_index=9,
    ),
    fixed_params={
        "batch_size": 2,
        "sequence_length": 256,
        #
        "y_samples": None,
    },
    key_fields={
        "finetuned_ckpt_uuid",
    },
    bindings=[
        scopes.ArgNameBindingSpec("fisher_type", "diagonal"),
        scopes.ArgNameBindingSpec("y_samples", None),
        #
        scopes.ArgNameBindingSpec("fisher_class_chunk_size", 4),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", target_tasks.finetuning_dataset),
        #
        scopes.ArgNameBindingSpec("hf_back_compat", False),
        scopes.ArgNameBindingSpec("pretrained_body_only", True),
        scopes.ArgNameBindingSpec("use_roberta_head", True),
        #
        scopes.ArgNameBindingSpec("all_variables_mergeable", True),
    ],
)
class FisherComputation_DAPT_HeadOnly_TargetTasks_LastCkpt_AllVars(ExperimentAbc):
    pass


###############################################################################
###############################################################################


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
        }


S2ORC_TASKS = ["cs", "bio_med"]

S2ORC_TASK_TO_DAPT_NAME = {
    "bio_med": "allenai/biomed_roberta_base",
    "cs": "allenai/cs_roberta_base",
}


@experiment.experiment(
    uuid="9c9ba7a921504e7e839e42208f14ef3c",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "trial_index": 0,
            "task": task,
            "pretrained_model": S2ORC_TASK_TO_DAPT_NAME[task],
            #
            "batch_size": 2,
            "y_samples": 1,
            "num_examples": 16384,
        }
        for task in S2ORC_TASKS
    ],
    fixed_params={
        "sequence_length": 256,
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
class FisherComputation_MlmS2orc_16384(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="5ace7e1e63ce47698a2f2eb9c63361f0",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "trial_index": 0,
            "task": task,
            "pretrained_model": S2ORC_TASK_TO_DAPT_NAME[task],
            #
            "batch_size": 2,
            "y_samples": 1,
            "num_examples": 131072,
        }
        for task in S2ORC_TASKS
    ],
    fixed_params={
        "sequence_length": 256,
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
class FisherComputation_MlmS2orc_131072(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="aa443de7ca354b75b50d8e3e86c48ca9",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "trial_index": 0,
            "task": task,
            "pretrained_model": S2ORC_TASK_TO_DAPT_NAME[task],
            #
            "batch_size": 2,
            "y_samples": 1,
            "num_examples": 1048576,
        }
        for task in S2ORC_TASKS
    ],
    fixed_params={
        "sequence_length": 256,
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
class FisherComputation_MlmS2orc_1048576(ExperimentAbc):
    pass


@experiment.experiment(
    uuid="6affb6791fc840ca902884b7a6d198e5",
    group=PaperExpGroup,
    params_cls=MlmFisherParams,
    executable_cls=fisher_execs.fisher_computation,
    varying_params=[
        {
            "trial_index": 0,
            "task": task,
            "pretrained_model": S2ORC_TASK_TO_DAPT_NAME[task],
            #
            "batch_size": 2,
            "y_samples": 1,
            "num_examples": 16384,
        }
        for task in S2ORC_TASKS
    ],
    fixed_params={
        "sequence_length": 256,
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
        #
        scopes.ArgNameBindingSpec("max_grad_value", 1.0),
    ],
)
class FisherComputation_MlmS2orc_16384_Clipped1(ExperimentAbc):
    pass
