"""TODO: Add title."""
import bert
import tensorflow as tf
import tensorflow_probability as tfp

from m251.data.processing.constants import NUM_GLUE_LABELS
from m251.models import model_abcs

from . import bert as bert_common


class BertExtractiveQa(tf.keras.Model, model_abcs.MergeableModel):
    def __init__(self, bert_layer, pad_token_id, **kwargs):
        super().__init__(**kwargs)
        self.bert_layer = bert_layer
        self.pad_token_id = pad_token_id

        self.custom_loss = extractive_qa_loss

        self.head = tf.keras.layers.Dense(
            units=2,
            activation=None,
            # kernel_regularizer=tf.keras.regularizers.l2(lmbda_head),
            name="extractive_qa_head",
        )

        self.regularizers = []

    @property
    def is_hf(self):
        return getattr(self.bert_layer, "is_hf", False)

    def call(self, x, training=None, mask=None):
        inputs = x["input_ids"]
        token_type_ids = x["token_type_ids"]

        out = self.bert_layer([inputs, token_type_ids], training=training, mask=mask)

        out = self.head(out, training=training)
        padding = tf.cast(tf.equal(inputs, self.pad_token_id), tf.float32)
        padding = tf.expand_dims(padding, -1)

        # Remove the masked out regions.
        out = out * (1 - padding) - 1e9 * padding

        # logits.shape = [2, batch, sequence_length]
        logits = tf.transpose(out, [2, 0, 1])

        return {
            "logits": logits,
        }

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.custom_loss(y, y_pred)
            for reg in self.regularizers:
                loss += reg(self)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return {"loss": loss}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def compute_task_logits(self, task_inputs, task, training=False):
        del task
        out = self(task_inputs, training=training)
        return out["logits"]

    def get_num_classes_for_task(self, task):
        return None

    ############################################

    def create_dummy_inputs(self, sequence_length):
        dummy_input = tf.keras.Input([sequence_length], dtype=tf.int32)
        return {
            "input_ids": dummy_input,
            "token_type_ids": dummy_input,
        }

    def load_pretrained_weights(self, pretrained_name, fetch_dir=None):
        bert_common.load_pretrained_weights(
            self.bert_layer, pretrained_name, fetch_dir=fetch_dir
        )

    def create_metrics(self):
        # NOTE: Maybe add some metrics here.
        raise NotImplementedError

    ############################################

    def assert_single_task(self):
        # We are are always single task.
        pass

    def get_mergeable_body(self):
        return self.bert_layer

    def get_mergeable_variables(self):
        return self.get_mergeable_body().trainable_weights

    def get_heads(self):
        return [self.head]

    def get_classifier_head(self):
        return self.head

    def set_classifier_heads(self, heads):
        assert len(heads) == 1
        self.head.set_weights(heads[0].get_weights())

    def add_regularizer(self, regularizer):
        self.regularizers.append(regularizer)

    ############################################

    def compute_logits(self, x, training=None, mask=None):
        self.assert_single_task()
        logits = self(x, training=training, mask=mask)
        return logits["logits"]

    def log_prob_of_y_samples(self, data, num_samples, training=None, mask=None):
        raise NotImplementedError


###############################################################################


def extractive_qa_loss(y_true, y_pred):
    start_positions = y_true["start_positions"]
    end_positions = y_true["end_positions"]

    start_logits, end_logits = tf.unstack(y_pred["logits"], axis=0)

    def loss(positions, logits):
        seqlen = tf.shape(logits)[-1]
        # Recall that positions will be 0 for an example if it has no answer.
        # Thus we are encouraging the model to favor a start and end of 0
        # in those cases, which corresponds to the CLS token.
        one_hot_positions = tf.one_hot(positions, depth=seqlen, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

    start_loss = loss(start_positions, start_logits)
    end_loss = loss(end_positions, end_logits)

    total_loss = (start_loss + end_loss) / 2.0
    return total_loss


###############################################################################


def get_untrained_bert(architecture, pad_token_id, fetch_dir=None):
    bert_layer = bert_common.get_bert_layer(architecture, fetch_dir=fetch_dir)
    return BertExtractiveQa(bert_layer, pad_token_id=pad_token_id)
