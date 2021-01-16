"""TODO: Add title.

TBH, this isn't really roberta, but rather using huggingface models.
"""
import itertools
import tensorflow as tf

from transformers import AutoTokenizer
from transformers import TFRobertaModel
from transformers import TFRobertaForSequenceClassification
from transformers import TFAutoModelForQuestionAnswering

_DRY = {
    TFRobertaModel: {
        "roberta-large",
        "roberta-base",
        #
        "allenai/cs_roberta_base",
        "allenai/biomed_roberta_base",
        "allenai/reviews_roberta_base",
        "allenai/news_roberta_base",
    },
    TFRobertaForSequenceClassification: {
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
    },
}

[]

ROBERTA_CHECKPOINTS = frozenset(*_DRY.values())

# NOTE: No entry here corresponds to TFRobertaModel.
#   TFRobertaModel = roberta
#   TFRobertaForSequenceClassification => .roberta, .classifier
#   TFAutoModelForQuestionAnswering => .roberta, .qa_outputs
#
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


class RobertaWrapper(tf.keras.Model):
    """Wrapper around HF roberta to be compatable with my stuff."""

    def __init__(self, model, pad_token=1, back_compat=True, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.pad_token = pad_token
        self.is_hf = True
        self.back_compat = back_compat

        if isinstance(model, TFRobertaModel):
            self.body = model
            self.head = None
            self.head_input_has_sequence_dim = not back_compat

        elif isinstance(model, TFRobertaForSequenceClassification):
            self.body = model.roberta
            self.head = model.classifier
            self.head_input_has_sequence_dim = True

        elif isinstance(model, TFAutoModelForQuestionAnswering):
            self.body = model.roberta
            self.head = model.qa_outputs
            self.head_input_has_sequence_dim = True

        else:
            raise ValueError(f"Unsupported auto model class: {model}")

        if self.head:
            self.head.trainable = False

    @property
    def params(self):
        return self.model.config

    def call(self, inputs, training=None, **kwargs):
        del kwargs
        input_ids, _ = inputs

        roberta_inputs = {
            "input_ids": input_ids,
            "attention_mask": tf.cast(
                tf.not_equal(input_ids, self.pad_token), tf.int32
            ),
        }

        # NOTE: I was a bit mistaken about what to use from the output. See the
        # documentation on the ouput of the model call at
        # https://huggingface.co/transformers/model_doc/roberta.html#tfrobertamodel.
        # I should be using the first item of the last_hidden_state rather than the
        # pooler output. I'm leaving the pooler_output in for back-compatability with
        # my saved models, but I should change this.
        last_hidden_state, pooler_output = self.body(roberta_inputs, training=training)
        if self.head_input_has_sequence_dim:
            return last_hidden_state
        else:
            return tf.expand_dims(pooler_output, axis=-2)


def get_pretrained_roberta(pretrained_model, roberta_back_compat=True):
    # NOTE: This will be pretrained unlike our analogous method for bert.
    AutoModelClass = CHECKPOINT_TO_AUTO_MODEL.get(pretrained_model, TFRobertaModel)
    model = AutoModelClass.from_pretrained(
        pretrained_model, from_pt=from_pt(pretrained_model)
    )
    return RobertaWrapper(model, back_compat=roberta_back_compat)
