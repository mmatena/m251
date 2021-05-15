"""TODO: Add title."""
import bert
import tensorflow as tf
import tensorflow_probability as tfp

try:
    from transformers.modeling_tf_roberta import TFRobertaClassificationHead
except ModuleNotFoundError:
    from transformers.models.roberta.modeling_tf_roberta import (
        TFRobertaClassificationHead,
    )

from m251.data.processing.constants import NUM_GLUE_LABELS
from m251.models import model_abcs

from . import bert as bert_common


class BertGlueClassifier(tf.keras.Model, model_abcs.MergeableModel):
    def __init__(
        self,
        bert_layer,
        tasks,
        use_roberta_head=False,
        all_variables_mergeable=False,
        freeze_body=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bert_layer = bert_layer
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.num_classes = [NUM_GLUE_LABELS[t] for t in tasks]
        self.use_roberta_head = use_roberta_head
        self.freeze_body = freeze_body

        self.all_variables_mergeable = all_variables_mergeable

        if self.is_hf and getattr(bert_layer, "head", None) and len(tasks) == 1:
            # In this case, we are loading a pretrained classifier with its
            # pretrained head.
            self.heads = [bert_layer.head]

        elif use_roberta_head:
            self.heads = []
            for class_count in self.num_classes:
                config = bert_layer.params
                config_copy = config.from_dict(config.to_dict())
                setattr(config_copy, "num_labels", class_count)
                self.heads.append(TFRobertaClassificationHead(config_copy))

        else:
            self.heads = [
                tf.keras.layers.Dense(
                    units,
                    activation=None,
                    name=f"classifier_head_{i}",
                )
                for i, units in enumerate(self.num_classes)
            ]

        if self.freeze_body:
            if hasattr(self.bert_layer, "freeze"):
                self.bert_layer.freeze()
            else:
                self.bert_layer.trainable = False

        # TODO: There is probably a better way to get this custom loss working.
        self.custom_loss = _make_multitask_loss(
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            num_tasks=self.num_tasks,
        )

        self.regularizers = []

    @property
    def is_hf(self):
        return getattr(self.bert_layer, "is_hf", False)

    @property
    def head_input_has_sequence_dim(self):
        if self.use_roberta_head:
            return True
        return getattr(self.bert_layer, "head_input_has_sequence_dim", False)

    def _get_cls_representation_from_body_output(self, out):
        if self.head_input_has_sequence_dim:
            # NOTE: We now assume that we use the representation of the CLS token for classification.
            # Beaware that some models might pool all of the hidden states instead, but I do not
            # believe any of the models at the time of this writing are like that.
            return out[..., :1, :]
        else:
            return out[..., 0, :]

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
        all_out = self._get_cls_representation_from_body_output(all_out)

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

    @tf.function
    def compute_task_logits(self, task_inputs, task, training=False):
        input_ids = task_inputs["input_ids"]
        token_type_ids = task_inputs["token_type_ids"]

        out = self.bert_layer([input_ids, token_type_ids], training=training)

        # Get the CLS token representation.
        out = self._get_cls_representation_from_body_output(out)

        task_index = self.tasks.index(task)
        task_head = self.heads[task_index]
        return task_head(out, training=training)

    def get_num_classes_for_task(self, task):
        return NUM_GLUE_LABELS[task]

    ############################################

    def create_dummy_inputs(self, sequence_length):
        inputs = {}
        dummy_input = tf.keras.Input([sequence_length], dtype=tf.int32)
        for i in range(self.num_tasks):
            inputs[f"task_{i}_input_ids"] = dummy_input
            inputs[f"task_{i}_token_type_ids"] = dummy_input
        return inputs

    def load_pretrained_weights(self, pretrained_name, fetch_dir=None):
        bert_common.load_pretrained_weights(
            self.bert_layer, pretrained_name, fetch_dir=fetch_dir
        )

    def create_metrics(self):
        return {
            f"task_{i}": tf.keras.metrics.SparseCategoricalAccuracy(name=f"{name}_acc")
            for i, name in enumerate(self.tasks)
        }

    ############################################

    def assert_single_task(self):
        assert len(self.num_classes) == 1

    def get_mergeable_body(self):
        if self.all_variables_mergeable:
            return self
        else:
            return self.bert_layer

    def get_mergeable_variables(self):
        return self.get_mergeable_body().trainable_weights

    def get_heads(self):
        return list(self.heads)

    def get_classifier_head(self):
        self.assert_single_task()
        return self.heads[0]

    def _get_dummy_head_input(self):
        if self.head_input_has_sequence_dim:
            return tf.keras.Input([1, self.bert_layer.params.hidden_size])
        else:
            return tf.keras.Input([self.bert_layer.params.hidden_size])

    def set_classifier_heads(self, heads):
        assert len(heads) == len(self.heads)
        dummy_head_input = self._get_dummy_head_input()
        for old, new in zip(self.heads, heads):
            new(dummy_head_input)
            old.set_weights(new.get_weights())

    def add_regularizer(self, regularizer):
        self.regularizers.append(regularizer)

    ############################################

    def compute_logits(self, x, training=None, mask=None):
        self.assert_single_task()
        logits = self(x, training=training, mask=mask)
        return logits[self.SINGLE_TASK_KEY]

    def log_prob_of_y_samples(self, data, num_samples, training=None, mask=None):
        x, _ = data
        logits = self.compute_logits(x, training=training, mask=mask)

        samples = tfp.distributions.Categorical(logits=logits).sample([num_samples])
        samples = tf.one_hot(samples, depth=tf.shape(logits)[-1])

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        log_probs = tf.einsum("bc,sbc->sb", log_probs, tf.cast(samples, tf.float32))

        return log_probs


def get_untrained_bert(
    architecture,
    tasks,
    fetch_dir=None,
    hf_back_compat=True,
    pretrained_body_only=False,
    use_roberta_head=False,
    all_variables_mergeable=False,
    freeze_body=False,
):
    bert_layer = bert_common.get_bert_layer(
        architecture,
        fetch_dir=fetch_dir,
        hf_back_compat=hf_back_compat,
        body_only=pretrained_body_only,
    )
    return BertGlueClassifier(
        bert_layer,
        tasks=tasks,
        use_roberta_head=use_roberta_head,
        all_variables_mergeable=all_variables_mergeable,
        freeze_body=freeze_body,
    )


def _make_multitask_loss(loss, num_tasks):
    def multitask_loss(y_true, y_pred):
        per_task_losses = [
            loss(y_true[f"task_{i}"], y_pred[f"task_{i}"]) for i in range(num_tasks)
        ]
        return tf.reduce_mean(per_task_losses)

    return multitask_loss
