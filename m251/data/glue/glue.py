"""TODO: Add title."""
from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.data import preprocessing as preprocessing_execs
from del8.executables.data import tfds as tfds_execs

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
        task_bindings = [
            ("task", task),
            ("dataset_name", f"glue/{task}"),
        ]
        with scopes.binding_by_name_scopes(task_bindings):
            ds = _tfds_dataset()
            ds = _preprocesser(ds)
            ds = _common_prebatch_processer(ds)
            datasets.append(ds)

    mixture = _mixer(datasets)
    mixture = _batcher(mixture)

    return mixture
