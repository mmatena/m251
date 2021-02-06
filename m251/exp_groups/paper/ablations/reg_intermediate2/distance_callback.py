"""TODO: Add title."""
from absl import logging
import tensorflow as tf

from del8.core import data_class
from del8.core.di import executable


@data_class.data_class()
class DistanceSummary(object):
    def __init__(self, sq_l2_per_step=()):
        pass


class _DistanceCallback(tf.keras.callbacks.Callback):
    def __init__(self, distance_saver):
        super().__init__()
        self.distance_saver = distance_saver

    def on_train_begin(self, logs=None):
        self.sq_l2_per_step = []
        self.og_weights = [tf.identity(w) for w in self.model.get_mergeable_variables()]

    def on_train_batch_end(self, batch, logs=None):
        trainable_weights = self.model.get_mergeable_variables()
        from_pt_l2 = [
            tf.reduce_sum(tf.square(w - og_w))
            for og_w, w in zip(self.og_weights, trainable_weights)
        ]
        self.sq_l2_per_step.append(float(tf.reduce_sum(from_pt_l2).numpy()))

    def on_train_end(self, logs=None):
        self.distance_saver.save_distances(self.sq_l2_per_step)


@executable.executable()
class distance_saver_callback:
    def save_distances(self, sq_l2_per_step):
        summary = DistanceSummary(sq_l2_per_step=sq_l2_per_step)
        self.storage.store_item(summary)

    def call(self, storage):
        self.storage = storage
        return _DistanceCallback(self)
