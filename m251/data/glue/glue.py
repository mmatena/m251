"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.data import preprocessing as preprocessing_execs
from del8.executables.data import tfds as tfds_execs

from m251.data.processing.constants import SUPER_GLUE_TASKS
from m251.data.processing import glue as glue_processing
from m251.data.processing import mixture
from m251.models.bert import bert as bert_common


@executable.executable(
    default_bindings={
        "tfds_dataset": tfds_execs.tfds_dataset,
        "preprocesser": glue_processing.glue_preprocessor,
        "common_prebatch_processer": preprocessing_execs.common_prebatch_processer,
        "mixer": mixture.no_shuffle_mixer,
        "batcher": preprocessing_execs.batcher,
        "tokenizer": bert_common.bert_tokenizer,
    }
)
def glue_finetuning_dataset(
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
):
    datasets = []
    for task in tasks:
        is_super_glue = task in SUPER_GLUE_TASKS
        basename = "super_glue" if is_super_glue else "glue"
        task_bindings = [
            ("task", task),
            ("dataset_name", f"{basename}/{task}"),
        ]
        with scopes.binding_by_name_scopes(task_bindings):
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


def _handle_mnli(tasks):
    new_tasks = []
    for task in tasks:
        if task != "mnli":
            new_tasks.append(task)
        else:
            new_tasks.append("mnli_matched")
            new_tasks.append("mnli_mismatched")
    return new_tasks


@executable.executable(
    default_bindings={
        "tfds_dataset": tfds_execs.tfds_dataset,
        "preprocesser": glue_processing.glue_preprocessor,
        "tokenizer": bert_common.bert_tokenizer,
    }
)
def glue_robust_evaluation_dataset(
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
    for task in _handle_mnli(tasks):
        is_super_glue = task in SUPER_GLUE_TASKS
        basename = "super_glue" if is_super_glue else "glue"

        dataset_name = f"{basename}/{task}"

        cache_key = (dataset_name, split, num_examples, batch_size)
        label_cache_key = cache_key + ("labels",)
        if cache_validation_datasets and cache_key in _ROBUST_VALIDATION_DATASET_CACHE:
            datasets[task] = _ROBUST_VALIDATION_DATASET_CACHE[cache_key]
            datasets[f"{task}_labels"] = _ROBUST_VALIDATION_DATASET_CACHE[
                label_cache_key
            ]
            continue

        task_bindings = [
            ("task", task),
            ("dataset_name", dataset_name),
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
