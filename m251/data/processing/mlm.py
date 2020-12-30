"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable


def convert_dataset_to_features(
    dataset,
    tokenizer,
    sequence_length,
    masked_lm_prob=0.15,
    max_predictions_per_seq=None,
    text_key="text",
):
    """Creates an MLM task.

    I'm pretty sure there are many slight differences here from the pretraining
    task of BERT or RoBERTa, but this is a genuine MLM task.
    """
    if max_predictions_per_seq is None:
        # We'll bound the number based on the length of each example later on.
        # This just means we'll never use a bound lower than that one.
        max_predictions_per_seq = sequence_length

    pad_token = tokenizer.pad_token_id
    cls_token = tokenizer.cls_token_id
    sep_token = tokenizer.sep_token_id
    unmaskable_tokens = tf.constant([0, cls_token, sep_token])
    mask_token = tokenizer.mask_token_id
    vocab_size = tokenizer.vocab_size

    def py_map_fn(text):
        text = tf.compat.as_str(text.numpy())

        # TODO: probably better to do a random slice if the examples are typically long here.
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=sequence_length,
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        input_ids = tf.constant(input_ids, dtype=tf.int32)
        token_type_ids = tf.constant(token_type_ids, dtype=tf.int32)
        return input_ids, token_type_ids

    def map_fn(example):
        input_ids, token_type_ids = tf.py_function(
            func=py_map_fn,
            inp=[example[text_key]],
            Tout=[tf.int32, tf.int32],
        )
        return input_ids, token_type_ids

    def get_maskable_mask(input_ids):
        maskable = tf.ones(tf.shape(input_ids), dtype=tf.bool)
        for token in unmaskable_tokens:
            ok_with_this_token = tf.not_equal(input_ids, token)
            maskable &= ok_with_this_token
        return maskable

    def get_mask_of_modifications_to_keep(modified, num_to_predict):
        # The modified is the mask of all positions that we plan to
        # modify. All are valid, and the only point of this function is
        # to make_sure we don't exceed num_to_predict

        # Again, we are assuming that the data set has not been batched yet.
        indices_of_modified = tf.where(modified)
        indices_of_modified = tf.squeeze(indices_of_modified, axis=-1)
        modified_indices_to_keep = tf.random.shuffle(indices_of_modified)[
            :num_to_predict
        ]
        # It gets returned to us in int64.
        modified_indices_to_keep = tf.cast(modified_indices_to_keep, tf.int32)
        return tf.scatter_nd(
            modified_indices_to_keep[..., None],
            tf.ones(tf.shape(modified_indices_to_keep), dtype=tf.bool),
            shape=tf.shape(modified),
        )

    def mask_fn(input_ids):
        # Make sure to pass the input_ids before padding it.
        num_to_predict = tf.minimum(
            max_predictions_per_seq,
            tf.maximum(
                1,
                tf.cast(
                    tf.round(
                        tf.cast(tf.shape(input_ids)[-1], tf.float32) * masked_lm_prob
                    ),
                    tf.int32,
                ),
            ),
        )

        # Ignoring that there are some tokens (e.g., [CLS] and [SEP]) that we can't mask, let's
        # set up our masks.
        modified = tf.less(tf.random.uniform(tf.shape(input_ids)), masked_lm_prob)
        not_masked_if_modified = tf.less(
            tf.random.uniform(tf.shape(input_ids)), 1.0 - 0.8
        )
        keep_original_if_modified_but_not_masked = tf.less(
            tf.random.uniform(tf.shape(input_ids)), 0.5
        )

        # Now we update our modified mask to take into account the unmaskable tokens.
        maskable = get_maskable_mask(input_ids)
        modified &= maskable

        replace_with_mask_token = modified & ~not_masked_if_modified
        keep_original = (
            modified & not_masked_if_modified & keep_original_if_modified_but_not_masked
        )
        replace_with_random_token = (
            modified
            & not_masked_if_modified
            & ~keep_original_if_modified_but_not_masked
        )

        modified = get_mask_of_modifications_to_keep(modified, num_to_predict)

        replace_with_mask_token = tf.cast(modified & replace_with_mask_token, tf.int32)
        keep_original = tf.cast(modified & keep_original, tf.int32)
        replace_with_random_token = tf.cast(
            modified & replace_with_random_token, tf.int32
        )

        modified = tf.cast(modified, tf.int32)
        # There's a small chance we'll sample some special characters, but let's ignore that for now.
        random_tokens = tf.random.uniform(
            tf.shape(input_ids), 1, vocab_size, dtype=tf.int32
        )

        tokens_to_predict = modified * input_ids
        input_ids = tf.reduce_sum(
            [
                input_ids * (1 - modified),
                mask_token * replace_with_mask_token,
                input_ids * keep_original,
                replace_with_random_token * random_tokens,
            ],
            axis=0,
        )
        return input_ids, tokens_to_predict

    def pad(x):
        padding_length = sequence_length - tf.shape(x)[-1]
        x = tf.concat([x, pad_token * tf.ones(padding_length, dtype=tf.int32)], axis=-1)
        # Ensure the shape is known as this is often needed for downstream steps.
        return tf.reshape(x, [sequence_length])

    def process_tokenized_fn(input_ids, token_type_ids):
        input_ids, tokens_to_predict = mask_fn(input_ids)

        # Zero-pad up to the sequence length.
        input_ids = pad(input_ids)
        tokens_to_predict = pad(tokens_to_predict)
        token_type_ids = pad(token_type_ids)

        x = {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
        }
        y = {
            # The tokens_to_predict will be non-zero at positions where we modified the input_ids.
            # The value will be the original value. We can do this as we will never mask padding
            # values. Even if we did, they shouldn't be part of the inputs and ignoring their
            # corresponding predictions would be a decent way of handling that data corruption.
            "tokens_to_predict": tokens_to_predict
        }
        return x, y

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(
        process_tokenized_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return dataset


###############################################################################


@executable.executable()
def mlm_preprocessor(dataset, tokenizer, sequence_length):
    return convert_dataset_to_features(
        dataset,
        tokenizer,
        sequence_length=sequence_length,
    )
