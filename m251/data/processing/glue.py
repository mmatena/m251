"""TODO: Add title."""
import collections

import numpy as np
import tensorflow as tf

from transformers.data.processors import glue as hf_glue

from del8.core.di import executable

from .constants import STSB_MIN, STSB_MAX, STSB_NUM_BINS

_glue_processors = hf_glue.glue_processors
_glue_output_modes = hf_glue.glue_output_modes


def _to_hf_task_name(task):
    if task == "stsb":
        task = "sts-b"
    elif task == "sst2":
        task = "sst-2"
    elif task == "mnli_matched":
        task = "mnli"
    elif task == "mnli_mismatched":
        task = "mnli-mm"
    return task


HackExample = collections.namedtuple("HackExample", ["text_a", "text_b", "label"])


class BoolQProcessor(object):
    def get_labels(self):
        return [0, 1]

    def get_example_from_tensor_dict(self, example):
        return HackExample(
            text_a=tf.compat.as_str(example["question"].numpy()),
            text_b=tf.compat.as_str(example["passage"].numpy()),
            label=example["label"].numpy(),
        )

    def tfds_map(self, example):
        return example


_glue_processors["boolq"] = BoolQProcessor
_glue_output_modes["boolq"] = "classification"


def convert_dataset_to_features(
    dataset,
    tokenizer,
    max_length,
    task,
    label_map_overrides=None,
    stsb_num_bins=STSB_NUM_BINS,
):
    """Note that this is only for single examples; won't work with batched inputs.

    Note that hugging face produces a dataset that's twice as fast when iterating over.
    This is because they process all of the examples initially and save them in a
    python array. However, that leads to huge upfront cost that can be minutes in the
    case of large datasets. However, this method has negligible upfront cost and iteration
    over the returned dataset probably won't be a bottleneck with models used in practice.
    """
    og_task = task
    task = _to_hf_task_name(task)
    pad_token = tokenizer.pad_token_id
    # NOTE: Not sure if this is correct, but it matches up for BERT. RoBERTa does
    # not appear to use token types.
    pad_token_segment_id = tokenizer.pad_token_type_id

    processor = _glue_processors[task]()
    output_mode = _glue_output_modes[task]

    if task == "sts-b":
        # STS-B regression.
        stsb_bins = np.linspace(STSB_MIN, STSB_MAX, num=stsb_num_bins + 1)
        stsb_bins = stsb_bins[1:-1]
    else:
        label_list = processor.get_labels()
        label_map = {label: i for i, label in enumerate(label_list)}

    def py_map_fn(keys, *values):
        example = {tf.compat.as_str(k.numpy()): v for k, v in zip(keys, values)}
        example = processor.get_example_from_tensor_dict(example)
        example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        input_ids = tf.constant(input_ids, dtype=tf.int32)
        token_type_ids = tf.constant(token_type_ids, dtype=tf.int32)

        if output_mode == "classification":
            label = label_map[example.label]
            if label_map_overrides and og_task in label_map_overrides:
                label = label_map_overrides[og_task][label]
            label = tf.constant(label, dtype=tf.int64)
        else:
            label = float(example.label)
            assert 0.0 <= label <= 5.0, f"Out of range STS-B label {label}."
            label = np.digitize(label, stsb_bins)
            label = tf.constant(label, dtype=tf.int64)
        return input_ids, token_type_ids, label

    def map_fn(example):
        input_ids, token_type_ids, label = tf.py_function(
            func=py_map_fn,
            inp=[list(example.keys()), *example.values()],
            Tout=[tf.int32, tf.int32, tf.int64],
        )
        return input_ids, token_type_ids, label

    def pad_fn(input_ids, token_type_ids, label):
        # Zero-pad up to the sequence length.
        padding_length = max_length - tf.shape(input_ids)[-1]

        input_ids = tf.concat(
            [input_ids, pad_token * tf.ones(padding_length, dtype=tf.int32)], axis=-1
        )
        token_type_ids = tf.concat(
            [
                token_type_ids,
                pad_token_segment_id * tf.ones(padding_length, dtype=tf.int32),
            ],
            axis=-1,
        )

        tf_example = {
            # Ensure the shape is known as this is often needed for downstream steps.
            "input_ids": tf.reshape(input_ids, [max_length]),
            "token_type_ids": tf.reshape(token_type_ids, [max_length]),
        }
        return tf_example, label

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


###############################################################################


@executable.executable()
def glue_preprocessor(
    dataset, task, tokenizer, sequence_length, glue_label_map_overrides=None
):
    return convert_dataset_to_features(
        dataset,
        tokenizer,
        max_length=sequence_length,
        task=task,
        label_map_overrides=glue_label_map_overrides,
    )
