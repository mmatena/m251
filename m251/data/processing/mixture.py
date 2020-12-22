"""TODO: Add title."""
import functools

import tensorflow as tf

from del8.core.di import executable


def mix_no_shuffle(datasets, example_keys=None, label_keys=None, prefix="task"):
    """From a list of tf.data.Datasets, create a single mixed dataset.

    NOTE: Right now the proportions of each dataset in the mixture will
    be uniform (and thus not dependent on dataset size.)

    NOTE: We assume all datasets have (example, label) tuples, where either
    entry can be a dict of tensors or a single tensor.
    """

    def key_by_task(x, index, keys):
        if not isinstance(x, dict):
            return {f"{prefix}_{index}": x}
        if keys is None:
            keys = x.keys()
        return {f"task_{index}_{key}": x[key] for key in keys}

    def key_by_task_fn(x, y, task_index):
        return {
            "x": key_by_task(x, task_index, example_keys),
            "y": key_by_task(y, task_index, label_keys),
        }

    def merge_zipped_fn(*args):
        x = {}
        y = {}
        for arg in args:
            x.update(arg["x"])
            y.update(arg["y"])
        return x, y

    datasets = [
        ds.map(
            functools.partial(key_by_task_fn, task_index=i),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        for i, ds in enumerate(datasets)
    ]

    return tf.data.Dataset.zip(tuple(datasets)).map(
        merge_zipped_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )


###############################################################################


@executable.executable()
def no_shuffle_mixer(datasets):
    return mix_no_shuffle(datasets)
