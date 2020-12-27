"""TODO: Add title.

NOTE: Everything here is from SimCLRv2. I'm using simclr to refer
to it for the sake of brevity.

NOTE: I am also only considering models without the selective kernels.
"""
import os
import shutil

from absl import logging

import tensorflow as tf

from del8.core.utils import tf_util

from . import resnet


# SimCLR deals only with images resized to this size.
IMAGE_SIZE = (224, 224)


_GC_PRETRAINED_BASE_PATH = "gs://simclr-checkpoints-tf2/simclrv2/pretrained"
_DEFAULT_FETCH_DIR = "~/.pretrained_simclrv2"


_PRETRAINED_MODELS_TO_PARAMS = {
    "r50_1x": (50, 1),
    "r50_2x": (50, 2),
    "r101_1x": (101, 1),
    "r101_2x": (101, 2),
    "r152_1x": (152, 1),
    "r152_2x": (152, 2),
}


def _copy_from_gcs(src, dst):
    basename = os.path.basename(src.rstrip("/"))
    if tf.io.gfile.isdir(src):
        subdst = os.path.join(dst, basename)
        os.mkdir(subdst)
        for item_name in tf.io.gfile.listdir(src):
            item_path = os.path.join(src, item_name)
            _copy_from_gcs(item_path, subdst)
    else:
        tf.io.gfile.copy(src, os.path.join(dst, basename))


def get_pretrained_simclr_checkpoint(model_name, fetch_dir=None):
    if fetch_dir is None:
        fetch_dir = os.path.expanduser(_DEFAULT_FETCH_DIR)
    else:
        fetch_dir = os.path.expanduser(fetch_dir)

    if not os.path.exists(fetch_dir):
        os.mkdir(fetch_dir)

    if model_name not in _PRETRAINED_MODELS_TO_PARAMS:
        raise ValueError(f"SimCLR model not found: {model_name}")

    full_model_name = f"{model_name}_sk0"
    fetch_path = os.path.join(_GC_PRETRAINED_BASE_PATH, full_model_name)

    local_dir = os.path.join(fetch_dir, full_model_name)
    if not os.path.exists(local_dir):
        logging.info(f"Downloading SimCLRv2 model from {fetch_path}")
        try:
            _copy_from_gcs(fetch_path, fetch_dir)
        except Exception as e:
            if os.path.exists(local_dir):
                shutil.rmtree(local_dir)
            raise e

    assert os.path.exists(local_dir)

    return os.path.join(local_dir, "saved_model")


def get_pretrained_simclr(model_name, image_size=IMAGE_SIZE, fetch_dir=None):
    saved_model_path = get_pretrained_simclr_checkpoint(model_name, fetch_dir=fetch_dir)
    # We suppress logs at warning or lower as loading the model generates a lot of
    # useless warnining logs.
    with tf_util.logging_level("ERROR"):
        with tf.device("cpu"):
            # We need to set compile=False or else we end up not being able to access the
            # trainable weights.
            saved_model = tf.keras.models.load_model(saved_model_path, compile=False)

    name_to_saved_weight = {
        v.name.replace("sync_batch_normalization", "batch_normalization"): v
        for v in saved_model.model.trainable_variables
    }

    depth, width_multiplier = _PRETRAINED_MODELS_TO_PARAMS[model_name]
    model = resnet.SimClrBaseModel(depth=depth, width_multiplier=width_multiplier)

    # Build the model.
    dummy_input = tf.keras.Input([*image_size, 3], dtype=tf.float32)
    model(dummy_input)

    for v in model.trainable_variables:
        name = v.name
        if name.startswith("sim_clr_base_model/"):
            name = name[len("sim_clr_base_model/") :]
        saved_weight = name_to_saved_weight[name]
        v.assign(saved_weight)

    return model
