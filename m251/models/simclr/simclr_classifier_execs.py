"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes
from del8.executables.models import checkpoints as ckpt_exec

from m251.models import model_execs
from . import simclr_classifier


@executable.executable(
    pip_packages=[
        "transformers==3.0.2",
    ],
)
def simclr_initializer(pretrained_model, tasks, fetch_dir=None):
    # NOTE: Unlike simclr, this will return a model that has been
    # initialized with the pretrained SimCLR weights.
    return simclr_classifier.get_initialized_simclr(
        pretrained_model, tasks=tasks, fetch_dir=fetch_dir
    )


@executable.executable()
def simclr_builder(model, tasks, image_size):
    # Also specific to GLUE classification.
    inputs = model.create_dummy_inputs(image_size=image_size)
    model(inputs)
    return model


@executable.executable(
    default_bindings={
        "model_checkpoint_loader": ckpt_exec.checkpoint_loader,
    }
)
def simclr_loader(model, _model_checkpoint_loader, checkpoint=None):
    # NOTE: Since our SimCLR initializer loads the pretrained weights upon initialization,
    # we do not load the pretrained weights here as they will be already loaded.
    if checkpoint:
        model = _model_checkpoint_loader(model, checkpoint=checkpoint)
    return model


@executable.executable(
    default_bindings={
        "initializer": simclr_initializer,
        "loader": simclr_loader,
        "builder": simclr_builder,
        "metrics": model_execs.multitask_classification_metrics,
    }
)
def simclr_finetuning_model(
    _initializer,
    _builder,
    _loader,
    _regularizer=None,
    _optimizer=None,
    _loss=None,
    _metrics=None,
):
    model = _initializer()

    # NOTE: Maybe add some way of binding a local variable via a closure so
    # that we don't have to do something like this just in case the returned
    # model is different the input model.
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
