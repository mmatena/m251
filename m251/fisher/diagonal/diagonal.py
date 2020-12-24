"""Code for computing a diagonal approximation to the Fisher."""
import tensorflow as tf

from del8.core.utils import hdf5_util

from .. import fisher_abcs


class DiagonalFisherComputer(fisher_abcs.FisherComputer):
    def __init__(self, model, total_examples, y_samples=None):
        super().__init__()

        self.model = model
        self.total_examples = total_examples
        self.y_samples = y_samples
        self.fisher_diagonals = [
            tf.Variable(tf.zeros(w.shape), trainable=False, name=f"fisher/{w.name}")
            for w in model.get_mergeable_variables()
        ]

    def train_step(self, data):
        if self.y_samples is None:
            return self.train_step_exact_y(data)
        else:
            return self.train_step_sample_y(data)

    @tf.function
    def train_step_exact_y(self, data):
        x, _ = data
        trainable_weights = self.model.get_mergeable_variables()

        with tf.GradientTape() as tape:
            logits = self.model.compute_logits(x, training=False)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
        probs = tf.nn.softmax(logits)  # [batch, num_classes]
        batch_size = tf.cast(tf.shape(probs)[0], tf.float32)

        grads = tape.jacobian(log_probs, trainable_weights)
        for g, fisher in zip(grads, self.fisher_diagonals):
            # g.shape = [batch, num_classes, *var.shape]
            update = tf.tensordot(probs, tf.square(g), [[0, 1], [0, 1]])
            fraction_of_total = batch_size / float(self.total_examples)
            fisher.assign_add(fraction_of_total * update)

        return {}

    @tf.function
    def train_step_sample_y(self, data):
        trainable_weights = self.model.get_mergeable_variables()

        with tf.GradientTape(persistent=True) as tape:
            # log_probs.shape = [y_samples, batch]
            log_probs = self.model.log_prob_of_y_samples(
                data, num_samples=self.y_samples, training=False
            )

            batch_size = tf.cast(tf.shape(log_probs)[1], tf.float32)

            for log_prob in log_probs:
                with tape.stop_recording():
                    grads = tape.jacobian(log_prob, trainable_weights)
                    for g, fisher in zip(grads, self.fisher_diagonals):
                        # g.shape = [batch, *var.shape]
                        update = tf.reduce_sum(tf.square(g), axis=0)
                        fraction_of_total = batch_size / tf.cast(
                            self.total_examples * self.y_samples, tf.float32
                        )
                        fisher.assign_add(fraction_of_total * update)

        return {}

    def get_fisher_matrix(self):
        return DiagonalFisherMatrix(self.fisher_diagonals)

    def get_original_model(self):
        return self.model


class DiagonalFisherMatrix(fisher_abcs.FisherMatrix):
    def __init__(self, fisher_diagonals):
        # NOTE: Be careful that these aren't set to trainable. The FisherComputer
        # and the loader sets them to trainable=False, so we shouldn't have an
        # issue with that in our normal use-case.
        self.fisher_diagonals = fisher_diagonals

    @classmethod
    def load(cls, file):
        return cls(hdf5_util.load_variables_from_hdf5(file, trainable=False))

    def save(self, file):
        hdf5_util.save_variables_to_hdf5(self.fisher_diagonals, file)
