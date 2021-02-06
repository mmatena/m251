"""TODO: Add title.

You should add a table that lists
    1) # of seconds for fine-tuning BERT base on RTE for 10 epochs on GPU X
    2) # of seconds for estimating the Fisher of BERT base on RTE on GPU X
    3) # of seconds for evaluating a single setting of lambda on N examples ...
    4) # of seconds for evaluating 50 settings of lambda or, do FLOPs.

"""
from absl import logging

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2_as_graph,
)

from del8.core.di import executable


def _get_flops(model, input_specs):
    """Credit goes to https://github.com/tensorflow/tensorflow/issues/32809#issuecomment-768977280"""
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(input_specs)
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name="")
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(
            graph=graph, run_meta=run_meta, cmd="op", options=opts
        )
        return flops.total_float_ops


@executable.executable()
def compute_eval_flops(compiled_model, sequence_length):
    input_specs = {
        "task_0_input_ids": tf.TensorSpec([1, sequence_length], dtype=tf.int32),
        "task_0_token_type_ids": tf.TensorSpec([1, sequence_length], dtype=tf.int32),
    }
    flops = _get_flops(compiled_model, input_specs)
    logging.info(f"FLOPS for evaluation on a single example: {flops}")


@executable.executable()
def compute_train_flops(compiled_model, sequence_length):
    x_specs = {
        "task_0_input_ids": tf.ones([1, sequence_length], dtype=tf.int32),
        "task_0_token_type_ids": tf.ones([1, sequence_length], dtype=tf.int32),
    }
    y_specs = {"task_0": tf.ones([1], dtype=tf.int32)}
    input_specs = [x_specs, y_specs]

    with tf.profiler.experimental.Profile("/tmp/tf2_profiler_logs"):
        compiled_model.train_step(input_specs)

    # logging.info(f"FLOPS for train on a single example: {flops}")


###############################################################################


def _num_params(d_model=768, n_layer=12, n_ctx=64):
    d_attn = d_model
    d_ff = 2 * d_model
    return 2 * d_model * n_layer * (2 * d_attn + d_ff)


def flops_fwd(d_model=768, n_layer=12, n_ctx=64):
    N = _num_params(d_model=d_model, n_layer=n_layer, n_ctx=n_ctx)
    return n_ctx * (2 * N + 2 * n_layer * n_ctx * d_model)


def flops_bkwd(d_model=768, n_layer=12, n_ctx=64):
    return 2 * flops_fwd(d_model=d_model, n_layer=n_layer, n_ctx=n_ctx)


def flops_train(d_model=768, n_layer=12, n_ctx=64):
    fwd = flops_fwd(d_model=d_model, n_layer=n_layer, n_ctx=n_ctx)
    bkwd = flops_bkwd(d_model=d_model, n_layer=n_layer, n_ctx=n_ctx)
    return fwd + bkwd


def flops_merge(n_models=2, d_model=768, n_layer=12, n_ctx=64):
    N = _num_params(d_model=d_model, n_layer=n_layer, n_ctx=n_ctx)
    M = n_models
    # Adds + multiplies + divs
    return 2 * N * (M - 1) + 2 * N * M + N


def flops_fisher(n_class=2, d_model=768, n_layer=12, n_ctx=64):
    fwd = flops_fwd(d_model=d_model, n_layer=n_layer, n_ctx=n_ctx)
    bkwd = flops_bkwd(d_model=d_model, n_layer=n_layer, n_ctx=n_ctx)
    return fwd + n_class * bkwd


RTE_TRAIN_EXAMPLES = 2_490
RTE_VALIDATION_EXAMPLES = 277
MERGE_POINTS = 50


# 547047132364800
# 5.5E+14
FT_10_EPOCHS_RTE_FLOPS = 10 * RTE_TRAIN_EXAMPLES * flops_train()
print("{:.1E}".format(FT_10_EPOCHS_RTE_FLOPS))

# 91174522060800
# 9.1E+13
RTE_FISHER_FLOPS = RTE_TRAIN_EXAMPLES * flops_fisher()
print("{:.1E}".format(RTE_FISHER_FLOPS))


# 4.0E+08
print("{:.1E}".format(flops_merge()))

# 2028937936896
# 2.0E+12
SINGLE_RTE_MERGE_APPLY = flops_merge() + RTE_VALIDATION_EXAMPLES * flops_fwd()
print("{:.1E}".format(SINGLE_RTE_MERGE_APPLY))

# 101446896844800
# 1.0E+14
ALL_RTE_MERGE_APPLY = MERGE_POINTS * SINGLE_RTE_MERGE_APPLY
print("{:.1E}".format(ALL_RTE_MERGE_APPLY))
