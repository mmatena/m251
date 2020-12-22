"""TODO: Add title."""
import bert
import tensorflow as tf

from m251.models import model_abcs

from . import bert as bert_common


class BertGlueClassifier(tf.keras.Model, model_abcs.MergeableModel):
    def __init__(self, bert_layer, tasks, **kwargs):
        super().__init__(**kwargs)
        self.bert_layer = bert_layer
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.num_classes = [_NUM_GLUE_LABELS[t] for t in tasks]

        self.heads = [
            tf.keras.layers.Dense(
                units,
                activation=None,
                # kernel_regularizer=tf.keras.regularizers.l2(lmbda_head),
                name=f"classifier_head_{i}",
            )
            for i, units in enumerate(self.num_classes)
        ]

        # TODO: There is probably a better way to get this custom loss working.
        self.custom_loss = _make_multitask_loss(
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            num_tasks=self.num_tasks,
        )

        self.regularizers = []

    def call(self, x, training=None, mask=None):
        inputs = [x[f"task_{i}_input_ids"] for i in range(self.num_tasks)]
        token_type_ids = [x[f"task_{i}_token_type_ids"] for i in range(self.num_tasks)]
        num_task_examples = [tf.shape(inpt)[0] for inpt in inputs]
        all_inputs = tf.concat(inputs, axis=0)
        all_token_type_ids = tf.concat(token_type_ids, axis=0)

        all_out = self.bert_layer(
            [all_inputs, all_token_type_ids], training=training, mask=mask
        )

        # Get the CLS token representation.
        all_out = all_out[..., 0, :]

        outs = tf.split(all_out, num_or_size_splits=num_task_examples, axis=0)

        return {
            f"task_{i}": head(out, training=training)
            for i, (out, head) in enumerate(zip(outs, self.heads))
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

        self.compiled_metrics.update_state(y, y_pred)

        ret = {"loss": loss}
        ret.update({m.name: m.result() for m in self.metrics})
        return ret

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    ############################################

    def create_dummy_inputs(self, sequence_length):
        inputs = {}
        dummy_input = tf.keras.Input([sequence_length], dtype=tf.int32)
        for i in range(self.num_tasks):
            inputs[f"task_{i}_input_ids"] = dummy_input
            inputs[f"task_{i}_token_type_ids"] = dummy_input
        return inputs

    def load_pretrained_weights(self, pretrained_name, fetch_dir=None):
        bert_ckpt = bert_common.get_pretrained_checkpoint(
            pretrained_name, fetch_dir=fetch_dir
        )
        bert.load_bert_weights(self.bert_layer, bert_ckpt)

    def create_metrics(self):
        return {
            f"task_{i}": tf.keras.metrics.SparseCategoricalAccuracy(name=f"{name}_acc")
            for i, name in enumerate(self.tasks)
        }

    ############################################

    def get_body(self):
        return self.bert_layer

    def get_heads(self):
        return list(self.heads)

    def get_head(self):
        assert len(self.heads) == 1
        return self.heads[0]

    def add_regularizer(self, regularizer):
        self.regularizers.append(regularizer)


def get_untrained_bert(architecture, tasks, fetch_dir=None):
    bert_layer = bert_common.get_bert_layer(architecture, fetch_dir=fetch_dir)
    return BertGlueClassifier(bert_layer, tasks=tasks)


def _make_multitask_loss(loss, num_tasks):
    def multitask_loss(y_true, y_pred):
        per_task_losses = [
            loss(y_true[f"task_{i}"], y_pred[f"task_{i}"]) for i in range(num_tasks)
        ]
        return tf.reduce_mean(per_task_losses)

    return multitask_loss


_NUM_GLUE_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mrpc": 2,
    "sst2": 2,
    "stsb": 1,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}
