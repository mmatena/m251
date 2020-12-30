"""TODO: Add title."""
import builtins

import bert
from bert import loader as tf2_bert_loader

import params_flow as pf

import tensorflow as tf
import tensorflow_probability as tfp

from m251.models import model_abcs

from . import bert as bert_common


class BertMlm(tf.keras.Model, model_abcs.FisherableModel):
    def __init__(
        self,
        bert_layer,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bert_layer = bert_layer
        self.bert_config = self.bert_layer.params

        assert not self.is_roberta, "TODO: Support RoBERTa MLM models."

        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        ns = f"{self.bert_layer.name}/cls/predictions/transform"
        self.dense = tf.keras.layers.Dense(
            units=self.hidden_size,
            # TODO: Add the intitializer.
            # kernel_initializer=self.create_initializer(),
            name=f"{ns}/dense",
        )

        self.layer_norm = pf.LayerNormalization(name=f"{ns}/LayerNorm")

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.output_bias = self.add_weight(
            f"{self.name}/{self.bert_layer.name}/cls/predictions/output_bias",
            shape=[self.vocab_size],
            initializer=tf.zeros_initializer(),
        )

    @property
    def is_roberta(self):
        return getattr(self.bert_layer, "is_roberta", False)

    @property
    def hidden_size(self):
        return self.bert_config.hidden_size

    @property
    def vocab_size(self):
        return self.bert_config.vocab_size

    @property
    def token_embeddings(self):
        assert not self.is_roberta, "TODO: Support RoBERTa MLM models."
        return self.bert_layer.embeddings_layer.word_embeddings_layer.embeddings

    def call(self, x, training=None, mask=None):
        input_ids = x["input_ids"]
        token_type_ids = x["token_type_ids"]

        out = self.bert_layer([input_ids, token_type_ids], training=training, mask=mask)
        out = self.dense(out, training=training)
        out = self.layer_norm(out, training=training)

        logits = tf.matmul(out, self.token_embeddings, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.output_bias)

        # Mask unused logits for safety.
        logits *= tf.cast(tf.not_equal(input_ids, 0)[..., None], tf.float32)

        return {
            "logits": logits,
            # Need to do this as keras is dumb with metrics sometimes.
            "tokens_to_predict": logits,
        }

    ############################################

    def get_mergeable_body(self):
        return self

    #############################################

    def compute_logits(self, x, training=None, mask=None):
        out = self(x, training=training, mask=mask)
        return out["logits"]

    def log_prob_of_y_samples(self, data, num_samples, training=None, mask=None):
        x, y = data
        logits = self.compute_logits(x, training=training, mask=mask)

        samples = tfp.distributions.Categorical(logits=logits).sample([num_samples])
        samples = tf.one_hot(samples, depth=tf.shape(logits)[-1])

        modified_tokens_mask = tf.not_equal(y["tokens_to_predict"], 0)
        modified_tokens_mask = tf.cast(modified_tokens_mask, tf.float32)

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        log_probs = tf.einsum(
            "bl,blc,sblc->sb",
            modified_tokens_mask,
            log_probs,
            tf.cast(samples, tf.float32),
        )

        return log_probs

    #############################################

    def create_dummy_inputs(self, sequence_length):
        dummy_input = tf.keras.Input([sequence_length], dtype=tf.int32)
        return {
            "input_ids": dummy_input,
            "token_type_ids": dummy_input,
        }

    def create_metrics(self):
        return {"tokens_to_predict": _mlm_average_nll}


###############################################################################


def get_untrained_bert(pretrained_model, fetch_dir=None):
    bert_layer = bert_common.get_bert_layer(pretrained_model, fetch_dir=fetch_dir)
    return BertMlm(bert_layer)


def _our_map_to_stock_variable_name(name, prefix="bert_mlm/bert"):
    real_name = name.split(":")[0]
    if real_name.startswith(f"{prefix}/cls/predictions"):
        # These are the MLM prediction variables that are in the BERT checkpoint but not in
        # the tf2 bert implementation.
        return real_name[len(prefix) + 1 :]
    return tf2_bert_loader.map_to_stock_variable_name(name, prefix)


def load_pretrained_weights(model, pretrained_model, fetch_dir=None):
    bert_ckpt = bert_common.get_pretrained_checkpoint(
        pretrained_model, fetch_dir=fetch_dir
    )

    # We have to do this ugly hack as the load_bert_weights method checks if the
    # model is an instance of BertModelLayer.
    old_isinstance = builtins.isinstance
    builtins.isinstance = (
        lambda x, y: True if y == bert.BertModelLayer else old_isinstance(x, y)
    )
    bert.load_bert_weights(model, bert_ckpt, _our_map_to_stock_variable_name)
    builtins.isinstance = old_isinstance

    return model


###############################################################################


def _mlm_average_nll(y_true, y_pred):
    # Negative log-liklihood average across predicted tokens. Lower values are better.
    logits = y_pred
    tokens_to_predict = y_true
    # Keras casts this into float32 for whatever reason.
    tokens_to_predict = tf.cast(tokens_to_predict, tf.int32)

    log_probs = tf.nn.log_softmax(logits)
    nlls = tf.gather(-log_probs, tokens_to_predict, batch_dims=2, axis=-1)

    mask = tf.cast(tf.not_equal(tokens_to_predict, 0), tf.float32)
    mask /= tf.maximum(1.0, tf.reduce_sum(mask, axis=-1, keepdims=True))
    return tf.einsum("bs,bs->b", nlls, mask)
