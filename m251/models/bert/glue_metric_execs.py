"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable

from del8.executables.evaluation import metrics

from m251.data.processing.constants import STSB_MIN, STSB_MAX, STSB_NUM_BINS


def _convert_stsb(output_classes):
    bin_width = (STSB_MAX - STSB_MIN) / STSB_NUM_BINS
    return tf.cast(output_classes, tf.float32) * bin_width + 0.5 * bin_width


def _stsb_pearson_corrcoef(targets, predictions, *args, **kwargs):
    targets = _convert_stsb(targets)
    predictions = _convert_stsb(predictions)
    return metrics.pearson_corrcoef(targets, predictions, *args, **kwargs)


def _stsb_spearman_corrcoef(targets, predictions, *args, **kwargs):
    targets = _convert_stsb(targets)
    predictions = _convert_stsb(predictions)
    return metrics.spearman_corrcoef(targets, predictions, *args, **kwargs)


_TASK_TO_METRICS = {
    "cola": [metrics.matthews_corrcoef],
    "mnli": [metrics.accuracy],
    "mnli_matched": [metrics.accuracy],
    "mnli_mismatched": [metrics.accuracy],
    "mrpc": [metrics.f1_score_with_invalid, metrics.accuracy],
    "sst2": [metrics.accuracy],
    "stsb": [_stsb_pearson_corrcoef, _stsb_spearman_corrcoef],
    "qqp": [metrics.f1_score_with_invalid, metrics.accuracy],
    "qnli": [metrics.accuracy],
    "rte": [metrics.accuracy],
    "wnli": [metrics.accuracy],
    #
    "boolq": [metrics.accuracy],
}


@executable.executable()
def glue_robust_metrics(tasks):
    return _TASK_TO_METRICS.copy()
