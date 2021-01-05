"""TODO: Add title.

- beta: 1e-6, 1e-7, 1e-8
- learning rate: 1e-2, 1e-3
- epochs: 1 to 16 (of 4096 examples regargless of value of num_examples)
- num_examples: 4096, 32768, 262144 (then maybe None: MNLI has 393k examples but won't get more than 262k with phase I trials)
    - Note that RTE has less than 4096, so I only need to do this for MNLI.
- variance_scaling? (probably not)
"""
import functools

from absl import logging

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.data import tfds as tfds_execs
from del8.executables.models import checkpoints
from del8.executables.training import optimizers

from m251.data.glue import glue
from m251.fisher.diagonal import variational_diagonal_execs as vardiag_exes
from m251.fisher.execs import fisher_execs

from m251.storage_util import eval_results

from ..group import BertMergingPrelimsGroup
from . import finetune_bert_base


@data_class.data_class()
class VariationalFisherParams(object):
    def __init__(
        self,
        #
        finetuned_exp_uuid,
        finetuned_run_uuid,
        finetuned_ckpt_uuid,
        #
        fisher_type,
        save_fisher_at_each_epoch,
        variational_fisher_beta,
        #
        learning_rate,
        num_examples,
        epochs,
        examples_per_epoch,
        #
        pretrained_model,
        task,
        #
        batch_size,
        sequence_length,
    ):
        pass

    def create_binding_specs(self):
        steps_per_epoch = self.examples_per_epoch // self.batch_size

        if self.fisher_type == "variational_diagonal":
            fisher_bindings = [
                scopes.ArgNameBindingSpec("fisher_type", self.fisher_type),
                scopes.ArgNameBindingSpec(
                    "compiled_fisher_computer",
                    vardiag_exes.variational_diag_fisher_computer,
                ),
                scopes.ArgNameBindingSpec(
                    "variational_fisher_beta", self.variational_fisher_beta
                ),
                scopes.ArgNameBindingSpec(
                    "save_fisher_at_each_epoch", self.save_fisher_at_each_epoch
                ),
            ]
        else:
            raise ValueError(f"Invalid fisher_type {self.fisher_type}.")

        return [
            scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
            scopes.ArgNameBindingSpec("tasks", [self.task]),
            #
            scopes.ArgNameBindingSpec("finetuned_exp_uuid", self.finetuned_exp_uuid),
            scopes.ArgNameBindingSpec("finetuned_run_uuid", self.finetuned_run_uuid),
            scopes.ArgNameBindingSpec("finetuned_ckpt_uuid", self.finetuned_ckpt_uuid),
            #
            scopes.ArgNameBindingSpec("num_examples", self.num_examples),
            scopes.ArgNameBindingSpec("sequence_length", self.sequence_length),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
            #
            scopes.ArgNameBindingSpec("learning_rate", self.learning_rate),
            #
            scopes.ArgNameBindingSpec("epochs", self.epochs),
            scopes.ArgNameBindingSpec("steps_per_epoch", steps_per_epoch),
        ] + fisher_bindings

    def create_preload_blob_uuids(self):
        return (self.finetuned_ckpt_uuid,)


@experiment.with_experiment_storages()
def create_varying_params_from_best_eval(
    exp,
    ft_exp,
    eval_exp,
    #
    dataset_sizes,
    betas,
    learning_rates,
    #
    tasks=frozenset(),
    reg_types=frozenset(),
    reg_strengths=frozenset(),
):
    run_uuids = ft_exp.retrieve_run_uuids(RunState.FINISHED)

    varying_params = []
    run_keys = []
    for run_uuid in run_uuids:
        run_params = ft_exp.retrieve_run_params(run_uuid)
        if (
            run_params.task not in tasks
            or run_params.reg_type not in reg_types
            or run_params.reg_strength not in reg_strengths
        ):
            continue
        # TODO: Make sure that no run keys are duplicated. Throw error now,
        # consider how to handle it in the future.
        run_key = ft_exp.retrieve_run_key(run_uuid)

        if any(runs.RunKey.has_same_values(rk, run_key) for rk in run_keys):
            # NOTE: I throw error now. In the future, I should think of what to do.
            # I'm thinking a first pass would be to manually remove the "bad" existing
            # one from the database. Also provide an option just to do it for each
            # duplicate. Also if I get this error, make sure it isn't being caused by
            # some misconfiguration.
            raise ValueError(
                f"Found duplicated run key {run_key.key_values} in the database. "
                "See comment in code above this raise statement for thoughts on what to do."
            )
        run_keys.append(run_key)

        eval_run_uuid = eval_results.get_eval_run_uuid_for_train_run(
            ft_exp, eval_exp, run_uuid
        )
        best = eval_results.get_best_result_for_run(
            eval_exp, eval_run_uuid, run_params.task
        )
        ckpt = best.checkpoint_blob_uuid

        for num_examples in dataset_sizes:
            for variational_fisher_beta in betas:
                for learning_rate in learning_rates:
                    varying_params.append(
                        {
                            "num_examples": num_examples,
                            "variational_fisher_beta": variational_fisher_beta,
                            "learning_rate": learning_rate,
                            "finetuned_exp_uuid": ft_exp.uuid,
                            "finetuned_run_uuid": run_uuid,
                            "finetuned_ckpt_uuid": ckpt,
                            "pretrained_model": run_params.pretrained_model,
                            "task": run_params.task,
                        }
                    )
    return varying_params


@experiment.experiment(
    uuid="5ecc49b0ccd843319284a2973cf9abbf",
    group=BertMergingPrelimsGroup,
    params_cls=VariationalFisherParams,
    executable_cls=fisher_execs.variational_fisher_computation,
    varying_params=functools.partial(
        create_varying_params_from_best_eval,
        ft_exp=finetune_bert_base.Glue_Regs,
        eval_exp=finetune_bert_base.GlueEval_Regs,
        #
        tasks=["rte"],
        reg_types=["iso"],
        reg_strengths=[0.0003],
        #
        dataset_sizes=[4096],
        betas=[1e-6, 1e-7, 1e-8],
        learning_rates=[1e-2, 1e-3],
    ),
    fixed_params={
        "fisher_type": "variational_diagonal",
        #
        "batch_size": 8,
        "sequence_length": 64,
        #
        "epochs": 16,
        "examples_per_epoch": 4096,
        #
        "save_fisher_at_each_epoch": True,
    },
    key_fields={
        "finetuned_run_uuid",
        "finetuned_ckpt_uuid",
        #
        "fisher_type",
        #
        "variational_fisher_beta",
        "learning_rate",
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
    ],
)
class RteBestCkpt_Iso_0003_PhaseI(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )

    def create_preload_blob_uuids(self, params):
        return params.create_preload_blob_uuids()


@experiment.experiment(
    uuid="1db11bdaa6ce4ee7b8ddeb8e1829da0d",
    group=BertMergingPrelimsGroup,
    params_cls=VariationalFisherParams,
    executable_cls=fisher_execs.variational_fisher_computation,
    varying_params=functools.partial(
        create_varying_params_from_best_eval,
        ft_exp=finetune_bert_base.Glue_Regs,
        eval_exp=finetune_bert_base.GlueEval_Regs,
        #
        tasks=["mnli"],
        reg_types=["iso"],
        reg_strengths=[0.0003],
        #
        dataset_sizes=[4096, 32768, 262144],
        betas=[1e-6, 1e-7, 1e-8],
        learning_rates=[1e-2, 1e-3],
    ),
    fixed_params={
        "fisher_type": "variational_diagonal",
        #
        "batch_size": 8,
        "sequence_length": 64,
        #
        "epochs": 16,
        "examples_per_epoch": 4096,
        #
        "save_fisher_at_each_epoch": True,
    },
    key_fields={
        "finetuned_run_uuid",
        "finetuned_ckpt_uuid",
        #
        "fisher_type",
        #
        "variational_fisher_beta",
        "learning_rate",
        "num_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec("dataset", glue.glue_finetuning_dataset),
        scopes.ArgNameBindingSpec("optimizer", optimizers.adam_optimizer),
    ],
)
class MnliBestCkpt_Iso_0003_PhaseI(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )

    def create_preload_blob_uuids(self, params):
        return params.create_preload_blob_uuids()
