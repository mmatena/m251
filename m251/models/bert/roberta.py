"""TODO: Add title.

TBH, this isn't really roberta, but rather using huggingface models.
"""
import itertools
import tensorflow as tf

from transformers import AutoTokenizer

from transformers import TFAutoModel
from transformers import TFAutoModelForSequenceClassification
from transformers import TFAutoModelForQuestionAnswering

from transformers import TFBertForSequenceClassification
from transformers import TFRobertaForSequenceClassification

from transformers import TFBertModel
from transformers import TFRobertaModel
from transformers.modeling_tf_utils import TFSequenceClassificationLoss
from transformers.modeling_tf_utils import TFQuestionAnsweringLoss


_DRY = {
    TFAutoModel: {
        "bert-base-uncased",
        #
        "roberta-large",
        "roberta-base",
        #
        "allenai/cs_roberta_base",
        "allenai/biomed_roberta_base",
        "allenai/reviews_roberta_base",
        "allenai/news_roberta_base",
    },
    TFAutoModelForSequenceClassification: {
        "roberta-large-mnli",
        #
        "textattack/roberta-base-RTE",
        "textattack/roberta-base-MNLI",
        #
        "textattack/bert-base-uncased-CoLA",
        "textattack/bert-base-uncased-MNLI",
        "textattack/bert-base-uncased-MRPC",
        "textattack/bert-base-uncased-SST-2",
        "textattack/bert-base-uncased-STS-B",
        "textattack/bert-base-uncased-QQP",
        "textattack/bert-base-uncased-QNLI",
        "textattack/bert-base-uncased-RTE",
    },
    TFAutoModelForQuestionAnswering: {
        "deepset/roberta-base-squad2",
        #
        "twmkn9/bert-base-uncased-squad2",
    },
}


ROBERTA_CHECKPOINTS = frozenset(itertools.chain(*_DRY.values()))


CHECKPOINT_TO_AUTO_MODEL = {
    task: auto_model for auto_model, tasks in _DRY.items() for task in tasks
}

del _DRY


def from_pt(pretrained_model):
    # NOTE: pt stands for "pytorch" NOT "pretrained".
    # Really just a heuristic that works for the models I'm supporting.
    return "/" in pretrained_model


def get_tokenizer(pretrained_model):
    return AutoTokenizer.from_pretrained(pretrained_model)


class HfWrapper(tf.keras.Model):
    """Wrapper around HF to be compatable with my stuff."""

    def __init__(self, model, pad_token, body_only=False, back_compat=True, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.pad_token = pad_token
        self.is_hf = True
        self.back_compat = back_compat
        self.force_use_pooled_output = False

        if isinstance(model, (TFBertModel, TFRobertaModel)):
            self.body = model
            self.head = None
            self.head_input_has_sequence_dim = False
            if isinstance(model, TFBertModel):
                self.force_use_pooled_output = True

        elif isinstance(model, TFSequenceClassificationLoss):
            self.body = model.layers[0]
            self.head = model.classifier
            if isinstance(model, TFBertForSequenceClassification):
                # https://huggingface.co/transformers/_modules/transformers/models/bert/modeling_tf_bert.html#TFBertForSequenceClassification
                self.head_input_has_sequence_dim = False
                self.force_use_pooled_output = True
            elif isinstance(model, TFRobertaForSequenceClassification):
                # https://huggingface.co/transformers/_modules/transformers/models/roberta/modeling_tf_roberta.html#TFRobertaForSequenceClassification
                self.head_input_has_sequence_dim = True
            else:
                raise ValueError(f"Unsupported auto model class: {model}")

        elif isinstance(model, TFQuestionAnsweringLoss):
            self.body, self.head = model.layers
            self.head_input_has_sequence_dim = True

        else:
            raise ValueError(f"Unsupported auto model class: {model}")

        if body_only:
            self.head = None
        elif self.head:
            self.head.trainable = False

    @property
    def params(self):
        return self.model.config

    def freeze(self):
        self.body.trainable = False

    def call(self, inputs, training=None, **kwargs):
        del kwargs
        input_ids, token_type_ids = inputs

        hf_inputs = {
            "input_ids": input_ids,
            "attention_mask": tf.cast(
                tf.not_equal(input_ids, self.pad_token), tf.int32
            ),
        }
        if not self.back_compat:
            # NOTE: I'm not sure of the significance of this.
            hf_inputs["token_type_ids"] = token_type_ids

        # NOTE: I was a bit mistaken about what to use from the output. See the
        # documentation on the ouput of the model call at
        # https://huggingface.co/transformers/model_doc/roberta.html#tfrobertamodel.
        # I should be using the first item of the last_hidden_state rather than the
        # pooler output. I'm leaving the pooled_output in for back-compatability with
        # my saved models, but I should change this.
        last_hidden_state, pooled_output = self.body(hf_inputs, training=training)
        if self.head_input_has_sequence_dim:
            return last_hidden_state
        elif self.force_use_pooled_output or self.back_compat:
            return tf.expand_dims(pooled_output, axis=-2)
        else:
            # NOTE: I don't think this is necessarily needed, but the shape becomes
            # consistent with what we had before, so I'm putting here for consistency.
            return last_hidden_state[:, :1, :]


def get_pretrained_roberta(pretrained_model, hf_back_compat=True, body_only=False):
    # NOTE: This will be pretrained unlike our analogous method for bert.
    AutoModelClass = CHECKPOINT_TO_AUTO_MODEL[pretrained_model]
    model = AutoModelClass.from_pretrained(
        pretrained_model, from_pt=from_pt(pretrained_model)
    )
    tokenizer = get_tokenizer(pretrained_model)
    return HfWrapper(
        model,
        pad_token=tokenizer.pad_token_id,
        back_compat=hf_back_compat,
        body_only=body_only,
    )
