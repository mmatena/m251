"""Code for computing a unitwise approximation to the Fisher."""
from absl import logging
import tensorflow as tf

from del8.core.utils import hdf5_util
from del8.core.utils import model_util

from .. import fisher_abcs


class LayerVariables(object):
    def __init__(self, layer_path, variables):
        self._layer_path = layer_path
        self._variables = variables

    @property
    def layer_path(self):
        return self._layer_path

    @property
    def variables(self):
        return list(self._variables.values())

    @property
    def layer_name(self):
        return self._layer_path.split("/")[-1]

    def __getattr__(self, key):
        return self._variables[key]

    def __getitem__(self, key):
        return self._variables[key]

    def as_dict(self):
        return dict(self._variables)

    def dense_to_homogeneous_matrix(self):
        # Raises KeyError if this isn't a dense layer (technically doesn't have a kernel).
        if "bias" not in self._variables:
            return tf.identity(self.kernel)
        else:
            return tf.concat([self.kernel, tf.expand_dims(self.bias, -2)], axis=-2)

    def can_have_unitwise_fisher(self):
        # NOTE: This method is currently equivalent to the layer being a dense layer.
        return "kernel" in self._variables

    def create_unitwise_fisher(self):
        try:
            m, n = self.kernel.shape
        except KeyError:
            return None

        if "bias" in self._variables:
            m += 1

        return tf.Variable(
            tf.zeros([m, m, n]),
            trainable=False,
            name=f"fisher/{self._layer_path}",
        )

    def assign_from_homogeneous_matrix(self, mat):
        if "bias" in self._variables:
            kernel, bias = mat[..., :-1, :], mat[..., -1, :]
            self.kernel.assign(kernel)
            self.bias.assign(bias)
        else:
            self.kernel.assign(mat)

    ###################################

    @classmethod
    def create_layer_vars_dict(cls, variables):
        return {
            k: LayerVariables(k, v)
            for k, v in model_util.get_layer_path_to_variables(variables).items()
        }

    @classmethod
    def all_as_dicts(cls, layer_vars_dict):
        return {k: v.as_dict() for k, v in layer_vars_dict.items()}

    @classmethod
    def all_to_unitwise_fishers(cls, layer_vars_dict):
        return {
            layer_path: layer_vars.create_unitwise_fisher()
            for layer_path, layer_vars in layer_vars_dict.items()
            if layer_vars.can_have_unitwise_fisher()
        }

    ###############

    @classmethod
    def dict_to_homogeneous_matrix(cls, variables):
        return LayerVariables("", variables).dense_to_homogeneous_matrix()


###############################################################################


def estimate_number_of_parameters_for_unitwise_dense_layers(model):
    variables = model.get_mergeable_variables()
    layer_vars_dict = LayerVariables.create_layer_vars_dict(variables)

    count = 0

    for layer_vars in layer_vars_dict.values():
        try:
            w = layer_vars.dense_to_homogeneous_matrix()
        except KeyError:
            continue

        m, n = w.shape

        count += n * m ** 2

    return count


###############################################################################


class UnitwiseFisherComputer(fisher_abcs.FisherComputer):
    def __init__(
        self,
        model,
        total_examples,
        class_chunk_size=None,
        y_samples=None,
        fisher_device="gpu",
    ):
        super().__init__()
        assert (
            y_samples is None
        ), "TODO: Support computation of unitwise fisher using sampling."
        assert (
            class_chunk_size is None
        ), "TODO: Support computation of unitwise fisher with class chunk size."
        self.model = model
        self.total_examples = total_examples
        self.y_samples = y_samples
        self.class_chunk_size = class_chunk_size

        layer_vars_dict = LayerVariables.create_layer_vars_dict(
            model.get_mergeable_variables()
        )

        # Need to convert to nested dicts rather than custom objects to be
        # compatable with the methods in tensorflow/python/data/util/nest.py
        self.layer_vars_dict = LayerVariables.all_as_dicts(layer_vars_dict)

        with tf.device(fisher_device):
            self.unitwise_fishers = LayerVariables.all_to_unitwise_fishers(
                layer_vars_dict
            )

    def get_fisher_matrix(self):
        return UnitwiseFisherMatrix(self.unitwise_fishers)

    def get_original_model(self):
        return self.model

    def train_step(self, data):
        x, _ = data

        with tf.GradientTape(persistent=True) as tape:
            logits = self.model.compute_logits(x, training=False)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

        probs = tf.nn.softmax(logits)  # [batch, num_classes]
        batch_size = tf.cast(tf.shape(probs)[0], tf.float32)

        fraction_of_total = batch_size / float(self.total_examples)

        grads = tape.jacobian(log_probs, self.layer_vars_dict)
        for name_scope, fisher in self.unitwise_fishers.items():

            # g.shape = [batch, num_classes, *var.shape]
            grad = LayerVariables.dict_to_homogeneous_matrix(grads[name_scope])

            # TODO: Make sure this einsum is correct.
            update = tf.einsum("bcio,bcjo,bc->ijo", grad, grad, probs)

            fisher.assign_add(fraction_of_total * update)

        return {}


###############################################################################


def _fisher_var_to_dict_key(v):
    # Remove the "fisher/" prefix.
    key = "/".join(v.name.split("/")[1:])
    # Remove the ":0" suffix.
    key = key.split(":")[0]
    return key


class UnitwiseFisherMatrix(fisher_abcs.FisherMatrix):
    def __init__(self, unitwise_fishers):
        # NOTE: Be careful that these aren't set to trainable. The FisherComputer
        # and the loader sets them to trainable=False, so we shouldn't have an
        # issue with that in our normal use-case.
        self.unitwise_fishers = unitwise_fishers

    @classmethod
    def load(cls, file):
        variables = hdf5_util.load_variables_from_hdf5(file, trainable=False)
        unitwise_fishers = {_fisher_var_to_dict_key(v): v for v in variables}
        return cls(unitwise_fishers)

    def save(self, file):
        hdf5_util.save_variables_to_hdf5(self.unitwise_fishers.values(), file)


###############################################################################


def merge_models(
    merged_model,
    root_model,
    mergeable_models,
    weighting=None,
    single_task=True,
):
    # NOTE: We are going to have variables that were not merged. Handle this
    # making sure that they are diag merged, set to first model's values, or
    # dummy merged.

    # If single_task=True, then we only care about the score of the first model.
    if not weighting:
        weighting = len(mergeable_models) * [1.0]

    assert len(mergeable_models) == len(weighting)

    layer_vars_dicts = [
        LayerVariables.create_layer_vars_dict(mm.model.get_mergeable_variables())
        for mm in mergeable_models
    ]
    fishers = [mm.fisher_matrix for mm in mergeable_models]

    root_layer_vars_dict = LayerVariables.create_layer_vars_dict(
        root_model.get_mergeable_variables()
    )

    mergeable_variables = merged_model.get_mergeable_variables()
    layer_vars_dict = LayerVariables.create_layer_vars_dict(mergeable_variables)

    unmerged_variables = {v.ref(): v for v in mergeable_variables}

    for layer_name, layer_vars in layer_vars_dict.items():
        for var in layer_vars.variables:
            del unmerged_variables[var.ref()]

        root_layer = root_layer_vars_dict[layer_name]
        root_homog_mat = root_layer.dense_to_homogeneous_matrix()

        lhs, rhs = [], []
        for weight, dikt, fisher in zip(weighting, layer_vars_dicts, fishers):
            fisher_mat = weight * fisher.unitwise_fishers[layer_name]

            homog_mat = dikt[layer_name].dense_to_homogeneous_matrix()

            lhs.append(fisher_mat)
            rhs.append(tf.einsum("ijo,jo->io", fisher_mat, homog_mat - root_homog_mat))

        rhs = tf.reduce_sum(rhs, axis=0)
        lhs = tf.reduce_sum(lhs, axis=0)

        # We do this so that the units (i.e. outputs) are treated as batches
        # by tf.linalg.lstsq().
        lhs = tf.transpose(lhs, [2, 0, 1])
        rhs = tf.transpose(rhs)
        merged_homog_mat = tf.linalg.lstsq(lhs, rhs[..., None], fast=False)
        # Go back to our convention.
        merged_homog_mat = tf.transpose(tf.squeeze(merged_homog_mat, axis=-1))
        # Go from delta to true value.
        merged_homog_mat += root_homog_mat

        layer_vars.assign_from_homogeneous_matrix(root_homog_mat)

    #
    #
    #
    #
    # DO SOMETHING WITH unmerged_variables
    print(unmerged_variables)
    #
    #
    #
    #

    if single_task:
        heads = [mergeable_models[0].model.get_classifier_head()]
    else:
        heads = [m.model.get_classifier_head() for m in mergeable_models]
    merged_model.set_classifier_heads(heads)

    return merged_model
