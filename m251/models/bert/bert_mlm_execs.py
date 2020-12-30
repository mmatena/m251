"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.models import checkpoints as ckpt_exec

from . import bert_mlm


@executable.executable()
def bert_initializer(pretrained_model, fetch_dir=None):
    return bert_mlm.get_untrained_bert(pretrained_model, fetch_dir=fetch_dir)


@executable.executable()
def bert_builder(model, sequence_length):
    inputs = model.create_dummy_inputs(sequence_length=sequence_length)
    model(inputs)
    return model


@executable.executable()
def bert_pretrained_loader(model, pretrained_model, fetch_dir=None):
    return bert_mlm.load_pretrained_weights(
        model, pretrained_model, fetch_dir=fetch_dir
    )


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
def bert_mlm_metrics(model):
    return model.create_metrics()


@executable.executable(
    default_bindings={
        "initializer": bert_initializer,
        "loader": bert_loader,
        "builder": bert_builder,
        "metrics": bert_mlm_metrics,
    }
)
def bert_mlm_model(
    _initializer,
    _builder,
    _loader,
    _metrics=None,
):
    model = _initializer()

    with scopes.binding_by_name_scope("model", model):
        model = _builder(model)
        model = _loader(model)

        kwargs = {}
        if _metrics:
            kwargs["metrics"] = _metrics()

        model.compile(**kwargs)

    return model
