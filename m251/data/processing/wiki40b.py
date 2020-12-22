"""TODO: Add title."""
import tensorflow as tf


def to_paragraphs(ds, max_paragraphs=32):
    def map_fn(x):
        text = x["text"]
        # Paragraphs have this within them. Replace with a space.
        text = tf.strings.regex_replace(text, "_NEWLINE_", " ")
        lines = tf.strings.split(text, sep="\n")
        # The _START_PARAGRAPH_ is always folled by the paragraph in a
        # single line.
        keep_mask = tf.roll(tf.equal(lines, "_START_PARAGRAPH_"), shift=1, axis=0)
        paragraphs = tf.boolean_mask(lines, keep_mask)
        # Put a limit on the number of paragraphs per article that we use. This
        # is to keep an article with a huge number of paragraphs from messing
        # up our data.
        paragraphs = tf.random.shuffle(paragraphs)
        paragraphs = paragraphs[:max_paragraphs]
        return {"text": paragraphs}

    ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
    return ds
