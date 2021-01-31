"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.models import checkpoints as ckpt_exec

from . import bert as bert_common
from . import roberta
from . import roberta_mlm


@executable.executable()
def roberta_initializer(pretrained_model, tokenizer):
    # NOTE: This will be pretrained unlike our analogous method for bert.
    return roberta_mlm.RobertaMlm(
        roberta_mlm.get_pretrained_roberta(pretrained_model), tokenizer.pad_token_id
    )


@executable.executable()
def roberta_builder(model, sequence_length):
    inputs = model.create_dummy_inputs(sequence_length=sequence_length)
    model(inputs)
    return model


@executable.executable()
def roberta_pretrained_loader(model):
    # The model will already be loaded, so this is a noop.
    return model


@executable.executable(
    default_bindings={
        "bert_pretrained_loader": roberta_pretrained_loader,
        "model_checkpoint_loader": ckpt_exec.checkpoint_loader,
    }
)
def roberta_loader(model, _model_checkpoint_loader, checkpoint=None):
    if checkpoint:
        model = _model_checkpoint_loader(model, checkpoint=checkpoint)
    return model


@executable.executable()
def roberta_mlm_metrics(model):
    return model.create_metrics()


@executable.executable()
def roberta_mlm_loss(model):
    return model.create_loss()


@executable.executable(
    default_bindings={
        "tokenizer": bert_common.bert_tokenizer,
        "initializer": roberta_initializer,
        "loader": roberta_loader,
        "builder": roberta_builder,
        "metrics": roberta_mlm_metrics,
    }
)
def roberta_mlm_model(
    _initializer,
    _builder,
    _loader,
    _optimizer=None,
    _loss=None,
    _metrics=None,
):
    model = _initializer()

    with scopes.binding_by_name_scope("model", model):
        model = _builder(model)
        model = _loader(model)

        kwargs = {}
        if _optimizer:
            kwargs["optimizer"] = _optimizer()
        if _loss:
            kwargs["loss"] = _loss()
        if _metrics:
            kwargs["metrics"] = _metrics()

        model.compile(**kwargs)

    return model
