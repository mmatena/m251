"""TODO: Add title."""
import tensorflow as tf

from transformers.data.processors import glue as hf_glue

from del8.core.di import executable

glue_tasks_num_labels = hf_glue.glue_tasks_num_labels
glue_processors = hf_glue.glue_processors
glue_output_modes = hf_glue.glue_output_modes


def _to_hf_task_name(task):
    if task == "stsb":
        task = "sts-b"
    elif task == "sst2":
        task = "sst-2"
    return task


def convert_dataset_to_features(
    dataset, tokenizer, max_length, task, pad_token=0, pad_token_segment_id=0
):
    """Note that this is only for single examples; won't work with batched inputs.

    Note that hugging face produces a dataset that's twice as fast when iterating over.
    This is because they process all of the examples initially and save them in a
    python array. However, that leads to huge upfront cost that can be minutes in the
    case of large datasets. However, this method has negligible upfront cost and iteration
    over the returned dataset probably won't be a bottleneck with models used in practice.
    """
    task = _to_hf_task_name(task)

    processor = glue_processors[task]()
    label_list = processor.get_labels()
    output_mode = glue_output_modes[task]
    assert output_mode == "classification", "TODO: Handle the one float task."

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
        label = tf.constant(label_map[example.label], dtype=tf.int64)
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


@executable.executable(
    pip_packages=["transformers"],
)
def glue_preprocessor(dataset, task, tokenizer, sequence_length):
    return convert_dataset_to_features(
        dataset,
        tokenizer,
        max_length=sequence_length,
        task=task,
    )
