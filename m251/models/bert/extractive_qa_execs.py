"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes
from del8.executables.models import checkpoints as ckpt_exec

from . import bert as bert_common
from . import extractive_qa as eqa


@executable.executable(
    default_bindings={
        "tokenizer": bert_common.bert_tokenizer,
    }
)
def bert_initializer(pretrained_model, tokenizer, fetch_dir=None, hf_back_compat=True):
    return eqa.get_untrained_bert(
        pretrained_model,
        pad_token_id=tokenizer.pad_token_id,
        fetch_dir=fetch_dir,
        hf_back_compat=hf_back_compat,
    )


@executable.executable()
def bert_builder(model, sequence_length):
    inputs = model.create_dummy_inputs(sequence_length=sequence_length)
    model(inputs)
    return model


@executable.executable()
def bert_pretrained_loader(model, pretrained_model, fetch_dir=None):
    model.load_pretrained_weights(pretrained_name=pretrained_model, fetch_dir=fetch_dir)
    return model


@executable.executable(
    default_bindings={
        "bert_pretrained_loader": bert_pretrained_loader,
        "model_checkpoint_loader": ckpt_exec.checkpoint_loader,
    }
)
def bert_loader(
    model, _bert_pretrained_loader, _model_checkpoint_loader, checkpoint=None
):
    # Handles loading of pretrained checkpoints and our own saved checkpoints.
    if checkpoint:
        model = _model_checkpoint_loader(model, checkpoint=checkpoint)
    else:
        model = _bert_pretrained_loader(model)
    return model


@executable.executable()
def squad2_finetuning_metrics(model):
    return model.create_metrics()


@executable.executable(
    default_bindings={
        "initializer": bert_initializer,
        "loader": bert_loader,
        "builder": bert_builder,
        # NOTE: Uncomment if I actually get some metrics working.
        # "metrics": squad2_finetuning_metrics,
    }
)
def eqa_finetuning_model(
    _initializer,
    _builder,
    _loader,
    _regularizer=None,
    _optimizer=None,
    _loss=None,
    _metrics=None,
):
    model = _initializer()

    with scopes.binding_by_name_scope("model", model):
        model = _builder(model)
        model = _loader(model)
        if _regularizer:
            model = _regularizer(model)

        kwargs = {}
        if _optimizer:
            kwargs["optimizer"] = _optimizer()
        if _loss:
            kwargs["loss"] = _loss()
        if _metrics:
            kwargs["metrics"] = _metrics()

        model.compile(**kwargs)

    return model
