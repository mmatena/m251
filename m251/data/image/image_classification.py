"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes

from m251.data.processing import mixture

from del8.executables.data import preprocessing as preprocessing_execs
from del8.executables.data import tfds as tfds_execs

from m251.data.processing import image_preprocess

from .constants import IMAGE_CLASSIFICATION_VAL_SPLIT_NAME


###############################################################################


def _preprocess_for_classification(
    dataset,
    image_size,
    is_training=False,
    color_distort=True,
    test_crop=True,
    image_key="image",
    label_key="label",
):
    def map_fn(x):
        label = x[label_key]
        image = x[image_key]
        image = image_preprocess.preprocess_image(
            image,
            height=image_size[0],
            width=image_size[1],
            is_training=is_training,
            color_distort=color_distort,
            test_crop=test_crop,
        )
        return {"image": image}, label

    return dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)


@executable.executable()
def image_classification_preprocessor(
    dataset,
    image_size,
    split,
):
    is_training = split == "train"
    return _preprocess_for_classification(
        dataset,
        image_size=image_size,
        is_training=is_training,
        color_distort=False,
        test_crop=True,
    )


###############################################################################


@executable.executable(
    default_bindings={
        "tfds_dataset": tfds_execs.tfds_dataset,
        "preprocesser": image_classification_preprocessor,
        "common_prebatch_processer": preprocessing_execs.common_prebatch_processer,
        "mixer": mixture.no_shuffle_mixer,
        "batcher": preprocessing_execs.batcher,
    }
)
def simclr_finetuning_dataset(
    tasks,
    # NOTE: We can also create a class and have some of these as public
    # methods with a default implementation if the logic probably isn't
    # reusable. I don't think I support overidding an entire public method
    # yet, but it's something I plan to support.
    _tfds_dataset,
    _preprocesser,
    _common_prebatch_processer,
    _mixer,
    _batcher,
    split,
):
    datasets = []
    for task in tasks:
        task_bindings = [
            ("task", task),
            ("dataset_name", task),
            ("split", split),
        ]
        with scopes.binding_by_name_scopes(task_bindings):
            tfds_split = split
            if split != "train":
                tfds_split = IMAGE_CLASSIFICATION_VAL_SPLIT_NAME[task]
            with scopes.binding_by_name_scope("split", tfds_split):
                ds = _tfds_dataset()
            ds = _preprocesser(ds)
            ds = _common_prebatch_processer(ds)
            datasets.append(ds)

    mixture = _mixer(datasets)
    mixture = _batcher(mixture)

    return mixture
