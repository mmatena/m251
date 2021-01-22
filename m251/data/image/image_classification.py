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
    color_distort=False,
    test_crop=True,
):
    is_training = split == "train"
    return _preprocess_for_classification(
        dataset,
        image_size=image_size,
        is_training=is_training,
        color_distort=color_distort,
        test_crop=test_crop,
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
            if split == "validation":
                tfds_split = IMAGE_CLASSIFICATION_VAL_SPLIT_NAME[task]
            with scopes.binding_by_name_scope("split", tfds_split):
                ds = _tfds_dataset()
            ds = _preprocesser(ds)
            ds = _common_prebatch_processer(ds)
            datasets.append(ds)

    mixture = _mixer(datasets)
    mixture = _batcher(mixture)

    return mixture


###############################################################################


# (dataset_name, split, num_examples, batch_size) -> dataset
# (dataset_name, split, num_examples, batch_size, 'labels') -> labels
_ROBUST_VALIDATION_DATASET_CACHE = {}


def _extract_labels(dataset):
    labels = []
    for _, minibatch_labels in dataset:
        labels.append(minibatch_labels)
    return tf.concat(labels, axis=0)


def _data_to_batches_list(dataset):
    return list(dataset)


@executable.executable(
    default_bindings={
        "tfds_dataset": tfds_execs.tfds_dataset,
        "preprocesser": image_classification_preprocessor,
    }
)
def robust_evaluation_dataset(
    tasks,
    _tfds_dataset,
    _preprocesser,
    batch_size,
    num_examples=None,
    split="validation",
    cache_validation_datasets=True,
    cache_validation_batches_as_lists=False,
):
    datasets = {}
    for task in tasks:
        if split == "validation":
            split = IMAGE_CLASSIFICATION_VAL_SPLIT_NAME[task]

        cache_key = (task, split, num_examples, batch_size)
        label_cache_key = cache_key + ("labels",)
        if cache_validation_datasets and cache_key in _ROBUST_VALIDATION_DATASET_CACHE:
            datasets[task] = _ROBUST_VALIDATION_DATASET_CACHE[cache_key]
            datasets[f"{task}_labels"] = _ROBUST_VALIDATION_DATASET_CACHE[
                label_cache_key
            ]
            continue

        task_bindings = [
            ("task", task),
            ("dataset_name", task),
            ("split", split),
        ]
        with scopes.binding_by_name_scopes(task_bindings):
            ds = _tfds_dataset()
            if num_examples is not None:
                ds = ds.take(num_examples)
            ds = ds.cache()
            ds = _preprocesser(ds)
            ds = ds.batch(batch_size)
            labels = _extract_labels(ds)
            if cache_validation_batches_as_lists:
                ds = _data_to_batches_list(ds)

        datasets[task] = ds
        datasets[f"{task}_labels"] = labels

        if cache_validation_datasets:
            _ROBUST_VALIDATION_DATASET_CACHE[cache_key] = ds
            _ROBUST_VALIDATION_DATASET_CACHE[label_cache_key] = labels

    return datasets
