"""TODO: Add title.

Common @executables for models.
"""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes


@executable.executable()
def regularize_body_l2_from_initial(model, reg_strength=0.0):
    if not reg_strength:
        return model

    og_weights = [tf.identity(w) for w in model.get_mergeable_variables()]

    def regularizer(model_during_training):
        trainable_weights = model_during_training.get_mergeable_variables()
        from_pt_l2 = [
            tf.reduce_sum(tf.square(w - og_w))
            for og_w, w in zip(og_weights, trainable_weights)
        ]
        return reg_strength * tf.reduce_sum(from_pt_l2)

    model.add_regularizer(regularizer)

    return model


@executable.executable()
def multitask_classification_metrics(tasks):
    return {
        f"task_{i}": tf.keras.metrics.SparseCategoricalAccuracy(name=f"{name}_acc")
        for i, name in enumerate(tasks)
    }
