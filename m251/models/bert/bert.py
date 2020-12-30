"""TODO: Add title."""
import os

import bert
import params_flow as pf
from transformers import BertTokenizer

from del8.core.di import executable

from . import roberta


_DEFAULT_FETCH_DIR = "~/.pretrained_bert"

_BERT_MODELS_GOOGLE = {
    "tiny": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-2_H-128_A-2.zip",
    "mini": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-256_A-4",
    "small": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-4_H-512_A-8",
    "medium": "https://storage.googleapis.com/bert_models/2020_02_20/uncased_L-8_H-512_A-8",
    "base": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip",
    "large": "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip",
}

# NOTE: These are like 'roberta-large' and 'roberta-base', so there should be no overlap.
_ROBERTA_CHECKPOINTS = roberta.ROBERTA_CHECKPOINTS


def get_tokenizer(pretrained_model=None):
    if pretrained_model in _ROBERTA_CHECKPOINTS:
        return roberta.get_tokenizer(pretrained_model)
    # TODO: Change as different models use different tokenizers.
    return BertTokenizer.from_pretrained("bert-base-uncased")


def _get_google_bert_model(model_name, fetch_dir=None):
    if fetch_dir is None:
        fetch_dir = os.path.expanduser(_DEFAULT_FETCH_DIR)
    else:
        fetch_dir = os.path.expanduser(fetch_dir)
    fetch_url = _BERT_MODELS_GOOGLE[model_name]
    fetched_file = pf.utils.fetch_url(fetch_url, fetch_dir=fetch_dir)
    fetched_dir = pf.utils.unpack_archive(fetched_file)

    folder_name = os.path.basename(fetch_url)
    if folder_name.endswith(".zip"):
        folder_name = folder_name[:-4]

    if os.path.exists(os.path.join(fetched_dir, folder_name)):
        return os.path.join(fetched_dir, folder_name)
    else:
        return fetched_dir


def get_bert_layer(model_name, fetch_dir=None, name="bert"):
    if model_name in _ROBERTA_CHECKPOINTS:
        # NOTE: This will be pretrained unlike if we chose a bert model.
        return roberta.get_pretrained_roberta(model_name)

    model_dir = _get_google_bert_model(model_name, fetch_dir)
    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    bert_params.mask_zero = True
    l_bert = bert.BertModelLayer.from_params(bert_params, name=name)

    setattr(l_bert, "is_roberta", False)

    return l_bert


def get_pretrained_checkpoint(model_name, fetch_dir=None):
    model_dir = _get_google_bert_model(model_name, fetch_dir)
    model_ckpt = os.path.join(model_dir, "bert_model.ckpt")
    return model_ckpt


def load_pretrained_weights(bert_layer, model_name, fetch_dir=None):
    if getattr(bert_layer, "is_roberta", False):
        # The RoBERTa layer will already have the pretrained weights loaded.
        return bert_layer
    ckpt = get_pretrained_checkpoint(model_name, fetch_dir=fetch_dir)
    bert.load_bert_weights(bert_layer, ckpt)
    return bert_layer


###############################################################################


@executable.executable(
    pip_packages=[
        "bert-for-tf2",
        "transformers==3.0.2",
    ],
)
def bert_tokenizer(pretrained_model):
    return get_tokenizer(pretrained_model)
