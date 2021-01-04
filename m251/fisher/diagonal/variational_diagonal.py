"""

Taken from Task2Vec https://arxiv.org/pdf/1902.03545.pdf.

A lot of stuff copied/adapted from the code at https://github.com/awslabs/aws-cv-task2vec.
"""
import time
import datetime

from absl import logging

from bert.embeddings import PositionEmbeddingLayer

import params_flow as pf

import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops

from del8.core.utils import hdf5_util
from del8.core.utils import monkey_patching

from .. import fisher_abcs

og_reshape = tf.reshape


@tf.function
def _prepend_dims_to_rank(x, rank):
    # NOTE: Is a no-op if rank(x) >= rank.
    # for _ in range(rank - len(x.shape)):
    #     x = tf.expand_dims(x, 0)
    shape = tf.concat(
        [tf.ones([(rank - tf.rank(x))], dtype=tf.int32), tf.shape(x)], axis=0
    )
    # return tf.reshape(x, shape)
    return og_reshape(x, shape)


@tf.function
def _change_batch_dim(shape, batch_size):
    return tf.concat([[batch_size], shape[1:]], axis=0)


# Have to do this strange hack.
_POS_EMBEDDING_BASE_RANK = 2
_POS_EMBEDDING_MARKER_RANK = 11


@tf.function
def _mark_as_pos_embeddings(x):
    return _prepend_dims_to_rank(x, _POS_EMBEDDING_MARKER_RANK)


@tf.function
def _is_marked_pos_embeddings(x):
    return tf.rank(x) == _POS_EMBEDDING_MARKER_RANK


@tf.function
def _extract_marked_pos_embeddings(x):
    # NOTE: Not the best way, but I was getting errors trying to do a reshape
    # and whatnot.
    return tf.squeeze(x)


class VariationalDiagFisherComputer(fisher_abcs.FisherComputer):
    def __init__(self, model, beta=1e-8, variance_scaling=0.05):
        super().__init__()
        if getattr(model, "is_roberta", None):
            # We assume that we are dealing with a bert-tf2 model throughout this.
            raise ValueError("TODO: Variational diagonal merge with RoBERTa")

        self.model = model
        self.variance_scaling = variance_scaling

        self.monkey_patcher = None
        self.batch_size = tf.Variable(0, dtype=tf.int32, trainable=False)

        # self.sampling = False
        self.sampling = tf.Variable(False, trainable=False)
        self.beta = beta

        self.logvars = []
        self.ref_to_logvar = {}
        # Log of the squared lambdas.
        self.loglambda2s = []
        self.ref_to_loglambda2 = {}

        for w in model.get_mergeable_variables():
            self._initialize_for_variable(w)

    def _initialize_for_variable(self, w):
        # Initial ballpark estimate for optimal variance is the variance
        # of the weights in the kernel
        var = tf.square(tf.math.reduce_std(w))

        # Further scale down the variance by some factor
        logvar = tf.ones_like(w) * tf.math.log(var * self.variance_scaling + 1e-8)
        logvar = tf.Variable(logvar, name=f"logvar/{w.name}")
        self.logvars.append(logvar)
        self.ref_to_logvar[w.ref()] = logvar

        # Initial guess for lambda is the l2 norm of the weights
        loglambda2 = tf.math.log(tf.reduce_mean(tf.square(w) + 1e-8))
        loglambda2 = tf.Variable(loglambda2, name=f"loglambda2/{w.name}")
        self.loglambda2s.append(loglambda2)
        self.ref_to_loglambda2[w.ref()] = loglambda2

    def fit(self, *args, **kwargs):
        # TODO: Put the before/after in a context manager.
        self._before_train()
        with self.monkey_patcher:
            ret = super().fit(*args, **kwargs)
        self._after_train()
        return ret

    def _before_train(self):
        self.model.trainable = False

        # NOTE: Maybe I'll want to add an explicit check that all mergeable variables get
        # accounted for.
        self.monkey_patcher = monkey_patching.MonkeyPatcherContext()
        self.monkey_patcher.patch_method(
            tf.keras.layers.Dense, "call", self._dense_call
        )
        self.monkey_patcher.patch_method(
            tf.keras.layers.Embedding, "call", self._embedding_call
        )
        self.monkey_patcher.patch_method(
            pf.LayerNormalization, "call", self._layer_norm_call
        )
        self.monkey_patcher.patch_method(
            PositionEmbeddingLayer, "call", self._pos_embedding_call
        )
        # Need to override the reshape in this way to get position embedding noising to work.
        self.monkey_patcher.patch_method(tf, "reshape", self._tf_reshape)

    def _after_train(self):
        # TODO: Set the model trainability status back to what it originally was.
        self.monkey_patcher = None
        print("TODO: Set the model trainability status back to what it originally was.")

    @tf.function
    def _dense_call(self, og_call, layer, inputs, *args, **kwargs):
        # TODO: Support layers with no bias.
        kernel = layer.kernel
        bias = layer.bias

        # NOTE: We need to put this in two if statements as autograph is dumb.
        if kernel.ref() not in self.ref_to_logvar:
            # This can happen for non-mergeable variables, so we skip them.
            return og_call(layer, inputs, *args, **kwargs)
        if not self.sampling:
            return og_call(layer, inputs, *args, **kwargs)

        kernel_logvar = self.ref_to_logvar[kernel.ref()]
        bias_logvar = self.ref_to_logvar[bias.ref()]

        output = core_ops.dense(
            inputs,
            kernel,
            bias,
            activations.get(None),
            dtype=layer._compute_dtype_object,
        )

        output_var = core_ops.dense(
            tf.square(inputs) + 1e-2,
            tf.exp(kernel_logvar),
            tf.exp(bias_logvar),
            activations.get(None),
            dtype=layer._compute_dtype_object,
        )

        eps = tf.random.normal(tf.shape(output))
        output += tf.sqrt(output_var) * eps

        if layer.activation is not None:
            output = layer.activation(output)

        return output

    @tf.function
    def _embedding_call(self, og_call, layer, inputs, *args, **kwargs):
        embeddings = layer.embeddings

        # NOTE: We need to put this in two if statements as autograph is dumb.
        if embeddings.ref() not in self.ref_to_logvar:
            # This can happen for non-mergeable variables, so we skip them.
            return og_call(layer, inputs, *args, **kwargs)
        if not self.sampling:
            return og_call(layer, inputs, *args, **kwargs)

        embeddings_logvar = self.ref_to_logvar[embeddings.ref()]
        stddev = tf.exp(0.5 * embeddings_logvar)

        batch_size = tf.shape(inputs)[0]

        # TODO: Find a better way of doing all of this.
        output = tf.zeros_like(embedding_ops.embedding_lookup_v2(embeddings, inputs))
        for i in tf.range(batch_size):
            eps = tf.random.normal(tf.shape(embeddings))
            perturbed_embeddings = embeddings + stddev * eps

            mask = tf.one_hot(i, depth=batch_size)

            output_i = embedding_ops.embedding_lookup_v2(perturbed_embeddings, inputs)
            for _ in range(len(output_i.shape) - 1):
                mask = mask[..., None]
            output += output_i * mask

        return output

    @tf.function
    def _layer_norm_call(self, og_call, layer, inputs, *args, **kwargs):
        gamma = layer.gamma
        beta = layer.beta

        # NOTE: We need to put this in two if statements as autograph is dumb.
        if gamma.ref() not in self.ref_to_logvar:
            # This can happen for non-mergeable variables, so we skip them.
            return og_call(layer, inputs, *args, **kwargs)
        if not self.sampling:
            return og_call(layer, inputs, *args, **kwargs)

        gamma_logvar = _prepend_dims_to_rank(
            self.ref_to_logvar[gamma.ref()], len(inputs.shape)
        )
        beta_logvar = _prepend_dims_to_rank(
            self.ref_to_logvar[beta.ref()], len(inputs.shape)
        )

        batch_size = tf.shape(inputs)[0]

        noise_shape = _change_batch_dim(tf.shape(gamma_logvar), batch_size)
        eps_gamma = tf.random.normal(noise_shape)
        eps_beta = tf.random.normal(noise_shape)

        gamma = gamma + eps_gamma * tf.exp(0.5 * gamma_logvar)
        beta = beta + eps_beta * tf.exp(0.5 * beta_logvar)

        # From https://github.com/kpe/params-flow/blob/master/params_flow/normalization.py
        x = inputs

        mean, var = tf.nn.moments(x, axes=-1, keepdims=True)

        inv = gamma * tf.math.rsqrt(var + layer.params.epsilon)
        res = x * tf.cast(inv, x.dtype) + tf.cast(beta - mean * inv, x.dtype)

        return res

    @tf.function
    def _pos_embedding_call(self, og_call, layer, inputs, *args, **kwargs):
        et = layer.embedding_table

        # NOTE: We need to put this in two if statements as autograph is dumb.
        if et.ref() not in self.ref_to_logvar:
            # This can happen for non-mergeable variables, so we skip them.
            return og_call(layer, inputs, *args, **kwargs)
        if not self.sampling:
            return og_call(layer, inputs, *args, **kwargs)

        et_logvar = _prepend_dims_to_rank(self.ref_to_logvar[et.ref()], rank=3)

        batch_size = self.batch_size
        noise_shape = _change_batch_dim(tf.shape(et_logvar), batch_size)
        eps_et = tf.random.normal(noise_shape)

        et = et + eps_et * tf.exp(0.5 * et_logvar)

        # Taken from
        # https://github.com/kpe/bert-for-tf2/blob/406bcac0a620f42d72797e4d826996b717fa0532/bert/embeddings.py#L43
        seq_len = inputs

        assert_op = tf.compat.v2.debugging.assert_less_equal(
            seq_len, layer.params.max_position_embeddings
        )

        with tf.control_dependencies([assert_op]):
            # slice to seq_len
            pos_embeddings = tf.slice(et, [0, 0, 0], [-1, seq_len, -1])

        pos_embeddings = _mark_as_pos_embeddings(pos_embeddings)
        return pos_embeddings

    @tf.function
    def _tf_reshape(self, og_reshape, tensor, shape, *args, **kwargs):
        if not _is_marked_pos_embeddings(tensor):
            return og_reshape(tensor, shape, *args, **kwargs)
        batch_size = self.batch_size

        shape = [batch_size] + shape[1:]

        et = _extract_marked_pos_embeddings(tensor)
        return og_reshape(et, shape, *args, **kwargs)

    def train_step(self, data):
        x, _ = data

        self.batch_size.assign(tf.shape(x["task_0_input_ids"])[0])

        og_logits = self.model.compute_logits(x, training=False)
        og_probs = tf.nn.softmax(og_logits, axis=-1)

        with tf.GradientTape() as tape:
            self.sampling.assign(True)
            sampled_logits = self.model.compute_logits(x, training=False)
            self.sampling.assign(False)

            loss = tf.keras.losses.categorical_crossentropy(
                y_true=og_probs, y_pred=sampled_logits, from_logits=True
            )
            loss += self.beta * self._get_compression_loss()

        trainable_vars = self.logvars + self.loglambda2s
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}

    def _get_compression_loss(self):
        """Get the model loss function for hessian estimation.
        Compute KL divergence assuming a normal posterior and a diagonal normal prior p(w) ~ N(0, lambda**2 * I)
        (where lambda is selected independently for each layer and shared by all filters in the same layer).
        Recall from the paper that the optimal posterior q(w|D) that minimizes the training loss plus the compression lost
        is approximatively given by q(w|D) ~ N(w, F**-1), where F is the Fisher information matrix.
        """
        kls = []

        _iter = zip(
            self.model.get_mergeable_variables(), self.logvars, self.loglambda2s
        )
        for w, logvar, loglambda2 in _iter:
            k = tf.size(w, tf.float32)
            w_norm2 = tf.reduce_sum(tf.square(w)) * tf.exp(-loglambda2)
            logvar_sum = tf.reduce_sum(logvar)
            trace = tf.reduce_sum(tf.exp(logvar)) * tf.exp(-loglambda2)
            lambda2_cost = loglambda2 * k

            # Standard formula for KL divergence of two normal distributions
            # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback%E2%80%93Leibler_divergence
            kl = w_norm2 + trace + lambda2_cost - logvar_sum - k
            kls.append(kl)

        return tf.reduce_sum(kls)

    def get_fisher_matrix(self):
        return VariationalDiagFisherMatrix(self.logvars)

    def get_original_model(self):
        return self.model


class VariationalDiagFisherMatrix(fisher_abcs.FisherMatrix):
    def __init__(self, logvars):
        # NOTE: Be careful that these aren't set to trainable.
        self.logvars = logvars

    @classmethod
    def load(cls, file):
        return cls(hdf5_util.load_variables_from_hdf5(file, trainable=False))

    def save(self, file):
        hdf5_util.save_variables_to_hdf5(self.logvars, file)
