"""TODO: Add title."""
import tensorflow as tf
import tensorflow_probability as tfp

from m251.data.image.constants import IMAGE_CLASSIFICATION_NUM_LABELS
from m251.models import model_abcs

from . import simclr as simclr_common


class SimclrClassifier(tf.keras.Model, model_abcs.MergeableModel):
    def __init__(self, base, tasks, all_variables_mergeable=False, **kwargs):
        super().__init__(**kwargs)
        self.base = base

        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.num_classes = [IMAGE_CLASSIFICATION_NUM_LABELS[t] for t in tasks]

        self.all_variables_mergeable = all_variables_mergeable

        # TODO: There is probably a better way to get this custom loss working.
        self.custom_loss = _make_multitask_loss(
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            num_tasks=self.num_tasks,
        )

        self.heads = [
            tf.keras.layers.Dense(
                units,
                activation=None,
                name="dense",
            )
            for i, units in enumerate(self.num_classes)
        ]
        self.regularizers = []

    @property
    def is_hf(self):
        # NOTE: My code was written with NLP in mind, so we
        # check sometimes to see if something is RoBERTa.
        return False

    #############################################

    def call(self, x, training=None, mask=None):
        del mask
        inputs = [x[f"task_{i}_image"] for i in range(self.num_tasks)]
        num_task_examples = [tf.shape(inpt)[0] for inpt in inputs]
        all_inputs = tf.concat(inputs, axis=0)

        all_out = self.base(all_inputs, training=training)

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

    #############################################

    def create_dummy_inputs(self, image_size):
        inputs = {}
        dummy_input = tf.keras.Input([*image_size, 3], dtype=tf.float32)
        for i in range(self.num_tasks):
            inputs[f"task_{i}_image"] = dummy_input
        return inputs

    def create_metrics(self):
        return {
            f"task_{i}": tf.keras.metrics.SparseCategoricalAccuracy(name=f"{name}_acc")
            for i, name in enumerate(self.tasks)
        }

    #############################################
    #############################################

    def assert_single_task(self):
        assert len(self.num_classes) == 1

    def add_regularizer(self, regularizer):
        self.regularizers.append(regularizer)

    #############################################

    def get_classifier_head(self):
        self.assert_single_task()
        return self.heads[0]

    def get_classifier_heads(self):
        # Make a defensive copy of the heads list.
        return list(self.heads)

    def set_classifier_heads(self, heads):
        for old, new in zip(self.heads, heads):
            # NOTE: I think the new heads will have to built. I can either
            # do that here or make sure I do it before passing in the heads.
            # I'll hold off on anything now, but this is probably the issue
            # if you get an exception here.
            old.set_weights(new.get_weights())

    #############################################

    def get_mergeable_body(self):
        return self if self.all_variables_mergeable else self.base

    def get_mergeable_variables(self):
        return self.get_mergeable_body().variables

    #############################################

    def compute_logits(self, x, training=None, mask=None):
        del mask
        self.assert_single_task()
        logits = self(x, training=training)
        return logits[self.SINGLE_TASK_KEY]

    def log_prob_of_y_samples(self, data, num_samples, training=None, mask=None):
        del mask
        x, _ = data
        logits = self.compute_logits(x, training=training)

        samples = tfp.distributions.Categorical(logits=logits).sample([num_samples])
        samples = tf.one_hot(samples, depth=tf.shape(logits)[-1])

        log_probs = tf.nn.log_softmax(logits, axis=-1)
        log_probs = tf.einsum("bc,sbc->sb", log_probs, tf.cast(samples, tf.float32))

        return log_probs

    ############################################

    @tf.function
    def compute_task_logits(self, task_inputs, task, training=False):
        images = task_inputs["image"]

        out = self.base(images, training=training)

        task_index = self.tasks.index(task)
        task_head = self.heads[task_index]
        return task_head(out, training=training)

    def get_num_classes_for_task(self, task):
        return IMAGE_CLASSIFICATION_NUM_LABELS[task]


def get_initialized_simclr(
    model_name, tasks, fetch_dir=None, all_variables_mergeable=False
):
    base = simclr_common.get_pretrained_simclr(model_name, fetch_dir=fetch_dir)
    return SimclrClassifier(
        base, tasks=tasks, all_variables_mergeable=all_variables_mergeable
    )


def _make_multitask_loss(loss, num_tasks):
    def multitask_loss(y_true, y_pred):
        per_task_losses = [
            loss(y_true[f"task_{i}"], y_pred[f"task_{i}"]) for i in range(num_tasks)
        ]
        return tf.reduce_mean(per_task_losses)

    return multitask_loss
