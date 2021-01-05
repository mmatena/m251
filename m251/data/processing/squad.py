"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable


MAX_ANSWERS_VAL_SQUAD2 = 6


def _is_whitespace(c):
    return c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F


def _compute_char_to_word_offset(context):
    ctx_words = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in context:
        if _is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                ctx_words.append(c)
            else:
                ctx_words[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(ctx_words) - 1)
    return char_to_word_offset, ctx_words


def _compute_tok_indices(ctx_words, tokenizer):
    orig_to_tok_index = []
    all_ctx_tokens = []
    for (i, token) in enumerate(ctx_words):
        orig_to_tok_index.append(len(all_ctx_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            all_ctx_tokens.append(sub_token)

    return orig_to_tok_index, all_ctx_tokens


def _improve_answer_span(
    doc_tokens, input_start, input_end, tokenizer, orig_answer_text
):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return new_start, new_end

    return input_start, input_end


def convert_dataset_to_features(dataset, tokenizer, max_length, split):
    is_training = split == "train"

    pad_token = tokenizer.pad_token_id
    # NOTE: Not sure if this is correct, but it matches up for BERT. RoBERTa does
    # not appear to use token types.
    pad_token_segment_id = tokenizer.pad_token_type_id

    def py_map_fn(question, context, is_impossible, answer_starts, answer_texts):
        question = tf.compat.as_str(question.numpy())
        context = tf.compat.as_str(context.numpy())
        is_impossible = is_impossible.numpy()
        answer_starts = [s.numpy() for s in answer_starts]
        answer_texts = [tf.compat.as_str(t.numpy()) for t in answer_texts]

        char_to_word_offset, ctx_words = _compute_char_to_word_offset(context)
        orig_to_tok_index, all_ctx_tokens = _compute_tok_indices(ctx_words, tokenizer)

        inputs = tokenizer.encode_plus(
            question,
            context,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # TODO: Handle cases where the question is longer than the sequence length.
        ctx_offset = 1 + input_ids.index(tokenizer.sep_token_id)

        answer_tok_ranges = []
        for offset, orig_text in zip(answer_starts, answer_texts):
            answer_length = len(orig_text)
            # Both are 0-based and inclusive.
            start_position = char_to_word_offset[offset]
            end_position = char_to_word_offset[offset + answer_length - 1]

            # TODO: Ignore examples where answers are not in the text.
            # TODO: Ignore examples where answers aren't in the context displayed.

            tok_start = orig_to_tok_index[start_position]

            if end_position < len(ctx_words) - 1:
                tok_end = orig_to_tok_index[end_position + 1] - 1
            else:
                tok_end = len(all_ctx_tokens) - 1

            tok_start, tok_end = _improve_answer_span(
                all_ctx_tokens, tok_start, tok_end, tokenizer, orig_text
            )

            answer_tok_ranges.append([ctx_offset + tok_start, ctx_offset + tok_end])

        input_ids = tf.constant(input_ids, dtype=tf.int32)
        token_type_ids = tf.constant(token_type_ids, dtype=tf.int32)
        answer_tok_ranges = tf.constant(
            answer_tok_ranges, dtype=tf.int32, shape=[len(answer_tok_ranges), 2]
        )

        return input_ids, token_type_ids, answer_tok_ranges

    def map_fn(example):
        input_ids, token_type_ids, answer_tok_ranges = tf.py_function(
            func=py_map_fn,
            inp=[
                example["question"],
                example["context"],
                example["is_impossible"],
                example["answers"]["answer_start"],
                example["answers"]["text"],
            ],
            Tout=[tf.int32, tf.int32, tf.int32],
        )
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

        pad_to_len = 1 if is_training else MAX_ANSWERS_VAL_SQUAD2
        padding_length = pad_to_len - tf.shape(answer_tok_ranges)[0]

        # When training on question with no answer, we'll have a 0 as both the start
        # and end token index, which corresponds to the CLS token.
        answer_tok_ranges = tf.concat(
            [answer_tok_ranges, tf.zeros([padding_length, 2], dtype=tf.int32)], axis=0
        )
        answer_tok_ranges = tf.reshape(answer_tok_ranges, [pad_to_len, 2])

        if is_training:
            answer_tok_ranges = tf.squeeze(answer_tok_ranges, 0)

        tf_label = {
            "is_impossible": example["is_impossible"],
            # NOTE: Both start and end are inclusive and zero-based.
            "start_positions": answer_tok_ranges[..., 0],
            "end_positions": answer_tok_ranges[..., 1],
        }

        return tf_example, tf_label

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


###############################################################################


@executable.executable()
def squad_preprocessor(dataset, tokenizer, sequence_length, split):
    return convert_dataset_to_features(
        dataset,
        tokenizer,
        max_length=sequence_length,
        split=split,
    )
