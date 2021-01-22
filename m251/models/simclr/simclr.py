"""TODO: Add title.

NOTE: Everything here is from SimCLRv2. I'm using simclr to refer
to it for the sake of brevity.

NOTE: I am also only considering models without the selective kernels.


NOTE: ACTUALLY I CHANGED THIS TO SUPPORT V1 FINETUNED CHECKPOINTS INSTEAD!!!

"""
import re
import os
import shutil
import weakref

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


_GC_FINETUNED_BASE_PATH = "gs://simclr-checkpoints/simclrv1/transfer/self_supervised"
_FINETUNED_SIZE_TO_PARAMS = {
    # NOTE: Note completely sure about these values, especially the 4x.
    "1x": (50, 1),
    "4x": (50, 4),
}
_FINETUNED_TASKS = {
    "birdsnap",
    "caltech101_split1",
    "cifar10",
    "cifar100",
    "dtd_split1",
    "fgvc_aircraft",
    "food101",
    "oxford_102flowers",
    "oxford_pets",
    "stanford_cars",
    "sun397",
    "voc2007",
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


def _get_fetch_path(model_name):
    if model_name in _PRETRAINED_MODELS_TO_PARAMS:
        full_model_name = f"{model_name}_sk0"
        fetch_path = os.path.join(_GC_PRETRAINED_BASE_PATH, full_model_name)
        return fetch_path, full_model_name

    elif (
        model_name[-2:] in _FINETUNED_SIZE_TO_PARAMS
        and model_name[-3] == "_"
        and model_name[:-3] in _FINETUNED_TASKS
    ):
        size = model_name[-2:]
        task = model_name[:-3]

        fetch_path = os.path.join(_GC_FINETUNED_BASE_PATH, size, task)
        return fetch_path, model_name[:-3]

    else:
        raise ValueError(f"SimCLR model not found: {model_name}")


def _get_fetch_dir(model_name, fetch_dir):
    if fetch_dir is None:
        fetch_dir = os.path.expanduser(_DEFAULT_FETCH_DIR)
    else:
        fetch_dir = os.path.expanduser(fetch_dir)

    if model_name in _PRETRAINED_MODELS_TO_PARAMS:
        return os.path.join(fetch_dir, "pretrained_simclrv2")
    else:
        size = model_name[-2:]
        return os.path.join(fetch_dir, size)


def get_pretrained_simclr_checkpoint(model_name, fetch_dir=None):
    fetch_dir = _get_fetch_dir(model_name, fetch_dir)

    if not os.path.exists(fetch_dir):
        os.makedirs(fetch_dir)

    fetch_path, full_model_name = _get_fetch_path(model_name)

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
    if os.path.exists(os.path.join(local_dir, "hub")):
        return os.path.join(local_dir, "hub")
    else:
        return os.path.join(local_dir, "saved_model")


def _get_model_params(model_name):
    if model_name in _PRETRAINED_MODELS_TO_PARAMS:
        return _PRETRAINED_MODELS_TO_PARAMS[model_name]
    elif model_name[-2:] in _FINETUNED_SIZE_TO_PARAMS:
        return _FINETUNED_SIZE_TO_PARAMS[model_name[-2:]]


LOADED_HEAD_WEIGHTS_WEAK_MAP = weakref.WeakKeyDictionary()


def get_pretrained_simclr(model_name, image_size=IMAGE_SIZE, fetch_dir=None):
    # We need to do this to prevent the layer names from increasing upon sequential
    # calls to this function.
    tf.compat.v1.reset_default_graph()

    saved_model_path = get_pretrained_simclr_checkpoint(model_name, fetch_dir=fetch_dir)
    # We suppress logs at warning or lower as loading the model generates a lot of
    # useless warnining logs.
    with tf_util.logging_level("ERROR"):
        with tf.device("cpu"):
            if model_name in _PRETRAINED_MODELS_TO_PARAMS:
                # We need to set compile=False or else we end up not being able to access the
                # trainable weights.
                saved_model = tf.keras.models.load_model(
                    saved_model_path, compile=False
                )
                saved_variables = saved_model.model.variables
            else:
                saved_model = tf.saved_model.load(saved_model_path, tags=[])
                saved_variables = saved_model.variables

    name_to_saved_weight = {
        v.name.replace("sync_batch_normalization", "batch_normalization"): v
        for v in saved_variables
    }

    depth, width_multiplier = _get_model_params(model_name)
    model = resnet.SimClrBaseModel(
        depth=depth, width_multiplier=width_multiplier, finetune_layer=0
    )

    # Build the model.
    dummy_input = tf.keras.Input([*image_size, 3], dtype=tf.float32)
    model(dummy_input)

    unused_weights = set_simclr_variables(model.variables, name_to_saved_weight)
    LOADED_HEAD_WEIGHTS_WEAK_MAP[model] = unused_weights

    return model


def set_simclr_variables(variables, name_to_saved_weight):
    for v in variables:
        name = v.name
        if name.startswith("sim_clr_base_model/"):
            name = name[len("sim_clr_base_model/") :]

        if name not in name_to_saved_weight:
            name = "/".join(v.name.split("/")[-2:])

        saved_weight = name_to_saved_weight[name]
        v.assign(saved_weight)
        del name_to_saved_weight[name]
    return name_to_saved_weight
