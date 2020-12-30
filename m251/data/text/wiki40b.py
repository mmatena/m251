"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.data import preprocessing as preprocessing_execs
from del8.executables.data import tfds as tfds_execs

from m251.data.processing import mlm
from m251.data.processing import wiki40b as wiki40b_processing

from m251.models.bert import bert as bert_common


@executable.executable(
    default_bindings={
        "dataset_name": "wiki40b/en",
        "tfds_try_gcs": True,
        "tfds_dataset": tfds_execs.tfds_dataset,
        "preprocesser": wiki40b_processing.paragraphs_preprocessor,
        "mlm_preprocessor": mlm.mlm_preprocessor,
        "common_prebatch_processer": preprocessing_execs.common_prebatch_processer,
        "batcher": preprocessing_execs.batcher,
        "tokenizer": bert_common.bert_tokenizer,
    }
)
def wiki40b_mlm_dataset(
    _tfds_dataset,
    _preprocesser,
    _mlm_preprocessor,
    _common_prebatch_processer,
    _batcher,
):
    ds = _tfds_dataset()
    ds = _preprocesser(ds)
    ds = _mlm_preprocessor(ds)
    ds = _common_prebatch_processer(ds)
    ds = _batcher(ds)
    return ds
