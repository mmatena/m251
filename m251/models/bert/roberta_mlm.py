"""TODO: Add title."""
import tensorflow as tf
import tensorflow_probability as tfp
from transformers import TFRobertaForMaskedLM

from m251.models import model_abcs

from . import bert_mlm
from . import roberta


class RobertaMlm(tf.keras.Model, model_abcs.FisherableModel):
    def __init__(
        self,
        roberta_layer,
        pad_token=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.roberta_layer = roberta_layer
        self.bert_config = self.roberta_layer.config
        self.pad_token = pad_token

    @property
    def is_hf(self):
        return True

    @property
    def hidden_size(self):
        return self.bert_config.hidden_size

    @property
    def vocab_size(self):
        return self.bert_config.vocab_size

    def call(self, x, training=None, mask=None):
        del mask
        input_ids = x["input_ids"]

        roberta_inputs = {
            "input_ids": input_ids,
            "attention_mask": tf.cast(
                tf.not_equal(input_ids, self.pad_token), tf.int32
            ),
        }
        (logits,) = self.roberta_layer(roberta_inputs, training=training)

        return {
            "logits": logits,
            # Need to do this as keras is dumb with metrics sometimes.
            "tokens_to_predict": logits,
        }

    ############################################

    def get_mergeable_body(self):
        return self.roberta_layer.roberta

    #############################################

    def compute_logits(self, x, training=None, mask=None):
        out = self(x, training=training, mask=mask)
        return out["logits"]

    def log_prob_of_y_samples(self, data, num_samples, training=None, mask=None):
        x, y = data
        logits = self.compute_logits(x, training=training, mask=mask)

        samples = tfp.distributions.Categorical(logits=logits).sample([num_samples])
        samples = tf.one_hot(samples, depth=tf.shape(logits)[-1])

        modified_tokens_mask = tf.not_equal(y["tokens_to_predict"], self.pad_token)
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
        return {"tokens_to_predict": bert_mlm.mlm_average_nll}


###############################################################################


def get_pretrained_roberta(pretrained_model):
    # NOTE: This will be pretrained unlike our analogous method for bert.
    return TFRobertaForMaskedLM.from_pretrained(
        pretrained_model, from_pt=roberta.from_pt(pretrained_model)
    )
