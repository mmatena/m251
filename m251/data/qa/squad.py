"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.data import preprocessing as preprocessing_execs
from del8.executables.data import tfds as tfds_execs

from m251.data.processing import squad as squad_processing
from m251.data.processing import mixture
from m251.models.bert import bert as bert_common


@executable.executable(
    default_bindings={
        "dataset_name": "squad/v2.0",
        "tfds_dataset": tfds_execs.tfds_dataset,
        "preprocesser": squad_processing.squad_preprocessor,
        "common_prebatch_processer": preprocessing_execs.common_prebatch_processer,
        "batcher": preprocessing_execs.batcher,
        "tokenizer": bert_common.bert_tokenizer,
    }
)
def squad2_finetuning_dataset(
    _tfds_dataset,
    _preprocesser,
    _common_prebatch_processer,
    _batcher,
):
    ds = _tfds_dataset()
    ds = _preprocesser(ds)
    ds = _common_prebatch_processer(ds)
    ds = _batcher(ds)

    return ds
