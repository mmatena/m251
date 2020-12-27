"""TODO: Add title."""
import collections
import itertools
import functools

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import experiment
from del8.core.experiment import runs
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.data import tfds as tfds_execs

from m251.fisher.diagonal import diagonal_execs
from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.data.image import image_classification
from m251.models import model_execs
from m251.models.simclr import simclr
from m251.models.simclr import simclr_classifier_execs as sc_exe

from . import simclr_finetune_iso
from . import simclr_fisher_iso
from ..simclr_group import SimclrMergingPrelimsGroup


@data_class.data_class()
class ModelToMerge(object):
    def __init__(
        self,
        task,
        train_run_uuid,
        fisher_run_uuid,
        model_checkpoint_uuid,
        fisher_matrix_uuid,
        # NOTE: Following added from the BERT GLUE version.
        model_checkpoint_index,
    ):
        pass


def create_pair_weightings(num_weightings):
    denom = num_weightings + 1
    return [((i + 1) / denom, 1 - (i + 1) / denom) for i in range(num_weightings)]


@data_class.data_class()
class DiagMergePairIsoParams(object):
    def __init__(
        self,
        #
        models_to_merge,
        #
        num_weightings,
        #
        pretrained_model,
        #
        fisher_type,
        fisher_params,
        #
        image_size,
        validation_examples,
        batch_size,
    ):
        pass

    def get_checkpoint_to_fisher_matrix_uuid(self):
        return {
            m.model_checkpoint_uuid: m.fisher_matrix_uuid for m in self.models_to_merge
        }

    def create_binding_specs(self):
        if self.fisher_type == "diagonal":
            fisher_bindings = [
                scopes.ArgNameBindingSpec(
                    "mergeable_model",
                    diagonal_execs.diagonal_mergeable_model_from_checkpoint,
                ),
                scopes.ArgNameBindingSpec(
                    "model_merger", diagonal_execs.diagonal_model_merger
                ),
            ]
        else:
            raise ValueError(f"Invalid fisher_type {self.fisher_type}.")

        return [
            scopes.ArgNameBindingSpec(
                "checkpoint_to_fisher_matrix_uuid",
                self.get_checkpoint_to_fisher_matrix_uuid(),
            ),
            scopes.ArgNameBindingSpec(
                "weightings", create_pair_weightings(self.num_weightings)
            ),
            #
            scopes.ArgNameBindingSpec(
                "checkpoints", [m.model_checkpoint_uuid for m in self.models_to_merge]
            ),
            scopes.ArgNameBindingSpec("tasks", [m.task for m in self.models_to_merge]),
            #
            scopes.ArgNameBindingSpec("pretrained_model", self.pretrained_model),
            #
            scopes.ArgNameBindingSpec("num_examples", self.validation_examples),
            scopes.ArgNameBindingSpec("image_size", self.image_size),
            scopes.ArgNameBindingSpec("batch_size", self.batch_size),
        ] + fisher_bindings


def create_varying_pairwise_merge_params(exp, fisher_exp, train_exp):
    # Purpose of the context manager is so that we don't keep reconnecting to GCP.
    with fisher_exp.get_storage(), train_exp.get_storage():
        return _create_varying_pairwise_merge_params(exp, fisher_exp, train_exp)


def _create_varying_pairwise_merge_params(exp, fisher_exp, train_exp):
    train_run_uuids = train_exp.retrieve_run_uuids(RunState.FINISHED)
    fisher_run_uuids = fisher_exp.retrieve_run_uuids(RunState.FINISHED)

    train_run_params = {
        rid: train_exp.retrieve_run_params(rid) for rid in train_run_uuids
    }
    fisher_run_params = [
        fisher_exp.retrieve_run_params(rid) for rid in fisher_run_uuids
    ]

    grouping_to_params = collections.defaultdict(list)
    for fi_rid, fi_rp in zip(fisher_run_uuids, fisher_run_params):
        assert fi_rp.ft_exp_uuid == train_exp.uuid

        tr_rp = train_run_params[fi_rp.ft_run_uuid]

        grouping_key = {}
        grouping_key.update(fisher_exp.create_run_key_values(fi_rp))
        grouping_key.update(train_exp.create_run_key_values(tr_rp))
        del grouping_key["finetuned_run_uuid"]
        del grouping_key["finetuned_ckpt_uuid"]
        del grouping_key["task"]
        grouping_key = hashabledict(grouping_key)

        saved_fisher_matrix = fisher_exp.retrieve_single_item_by_class(
            fisher_execs.SavedFisherMatrix, fi_rid
        )

        model_to_merge = ModelToMerge(
            task=tr_rp.task,
            train_run_uuid=fi_rp.ft_run_uuid,
            fisher_run_uuid=fi_rid,
            model_checkpoint_uuid=fi_rp.finetuned_ckpt_uuid,
            fisher_matrix_uuid=saved_fisher_matrix.blob_uuid,
            model_checkpoint_index=fi_rp.finetuned_ckpt_index,
        )

        grouping_to_params[grouping_key].append(model_to_merge)

    varying_params = []
    for grouping_key, models in grouping_to_params.items():
        # Make sure the tasks in this grouping are unique.
        assert len(set(p.task for p in models)) == len(models)

        base_param = {
            "pretrained_model": grouping_key["pretrained_model"],
            "fisher_type": grouping_key["fisher_type"],
        }

        if grouping_key["fisher_type"] == "diagonal":
            base_param["fisher_params"] = {
                "y_samples": grouping_key["diagonal_y_samples"],
            }
        else:
            raise ValueError(f"Invalid fisher_type {grouping_key['fisher_type']}.")

        for pair in itertools.combinations(models, 2):
            varying_param = base_param.copy()
            varying_param["models_to_merge"] = tuple(pair)
            varying_params.append(varying_param)

    return varying_params


###############################################################################
###############################################################################


# Model r50_1x, checkpoint from 20k steps in.
@experiment.experiment(
    #
    #
    #
    #
    #
    #
    #
    #
    #
    # uuid="f1ebc160a93d49f99f6920bf1d5d95c6",
    uuid="__TEST__f1ebc160a93d49f99f6920bf",
    #
    #
    #
    #
    #
    #
    #
    #
    #
    group=SimclrMergingPrelimsGroup,
    params_cls=DiagMergePairIsoParams,
    executable_cls=merging_execs.merge_and_evaluate_from_checkpoints,
    varying_params=functools.partial(
        create_varying_pairwise_merge_params,
        train_exp=simclr_finetune_iso.FinetuneSimclrIso_r50_1x,
        fisher_exp=simclr_fisher_iso.SimclrFisherIso__r50_1x__ckpt_20k,
    ),
    fixed_params={
        "num_weightings": 15,
        "validation_examples": 4096,
        "image_size": simclr.IMAGE_SIZE,
        "batch_size": 32,
    },
    key_fields={
        "models_to_merge",
        "num_weightings",
        "pretrained_model",
        "fisher_type",
        "fisher_params",
        "validation_examples",
    },
    bindings=[
        scopes.ArgNameBindingSpec("split", "validation"),
        scopes.ArgNameBindingSpec("shuffle", False),
        scopes.ArgNameBindingSpec("repeat", False),
        #
        scopes.ArgNameBindingSpec("tfds_dataset", tfds_execs.gcp_tfds_dataset),
        scopes.ArgNameBindingSpec(
            "dataset", image_classification.simclr_finetuning_dataset
        ),
        #
        scopes.ArgNameBindingSpec("initializer", sc_exe.simclr_initializer),
        scopes.ArgNameBindingSpec("builder", sc_exe.simclr_builder),
        scopes.ArgNameBindingSpec(
            "metrics", model_execs.multitask_classification_metrics
        ),
    ],
)
class SimclrDiagMergeIso__r50_1x__ckpt_20k(object):
    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )
