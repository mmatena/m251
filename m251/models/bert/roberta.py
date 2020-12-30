"""TODO: Add title."""
import tensorflow as tf

from transformers import RobertaTokenizer, TFRobertaModel


ROBERTA_CHECKPOINTS = {
    "roberta-large",
    "roberta-base",
}


def get_tokenizer(pretrained_model):
    return RobertaTokenizer.from_pretrained(pretrained_model)


class RobertaWrapper(tf.keras.Model):
    """Wrapper around HF roberta to be compatable with my stuff."""

    def __init__(self, model, checkpoint_name, pad_token=1, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.pad_token = pad_token
        self.is_roberta = True
        self.checkpoint_name = checkpoint_name

    @property
    def params(self):
        return self.model.config

    def call(self, inputs, training=None, **kwargs):
        del kwargs
        input_ids, _ = inputs

        roberta_inputs = {
            "input_ids": input_ids,
            "attention_mask": tf.cast(
                tf.not_equal(input_ids, self.pad_token), tf.int32
            ),
        }

        # This assumes we are a classifier. If doing sequence-level stuff, then return
        # the first output.
        _, out = self.model(roberta_inputs, training=training)
        return tf.expand_dims(out, axis=-2)


def get_pretrained_roberta(pretrained_model):
    # NOTE: This will be pretrained unlike our analogous method for bert.
    model = TFRobertaModel.from_pretrained(pretrained_model)
    return RobertaWrapper(model, checkpoint_name=pretrained_model)
