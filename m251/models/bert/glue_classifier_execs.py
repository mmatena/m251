"""TODO: Add title."""
import tensorflow as tf

from del8.core.di import executable
from del8.core.di import scopes
from del8.executables.models import checkpoints as ckpt_exec

from . import glue_classifier as bert_gc


@executable.executable(
    pip_packages=[
        "bert-for-tf2",
        "transformers",
    ],
)
def bert_initializer(pretrained_model, tasks, fetch_dir=None):
    return bert_gc.get_untrained_bert(
        pretrained_model, tasks=tasks, fetch_dir=fetch_dir
    )


@executable.executable()
def bert_builder(model, tasks, sequence_length):
    # Also specific to GLUE classification.
    inputs = model.create_dummy_inputs(sequence_length=sequence_length)
    model(inputs)
    return model


@executable.executable(
    pip_packages=[
        "bert-for-tf2",
    ],
)
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


# @executable.executable(
#     default_bindings={
#         "bert_pretrained_loader": bert_pretrained_loader,
#         "initializer": bert_initializer,
#         "builder": bert_builder,
#     }
# )
# @executable.executable()
# def pretrained_body_provider(
#     _initializer,
#     _builder,
#     _bert_pretrained_loader
# ):
#     # NOTE: Maybe not the most efficient way of doing this, but it works.
#     model = _initializer()
#     with scopes.binding_by_name_scope("model", model):
#         model = _builder(model)
#         model = _bert_pretrained_loader(model)
#         return model.get_mergeable_body()


@executable.executable()
def regularize_body_l2_from_initial(model, reg_strength=0.0):
    if not reg_strength:
        return model

    og_weights = [tf.identity(w) for w in model.get_mergeable_variables()]

    def regularizer(model_during_training):
        trainable_weights = model_during_training.get_mergeable_variables()
        from_pt_l2 = [
            tf.reduce_sum(tf.square(w - og_w))
            for og_w, w in zip(og_weights, trainable_weights)
        ]
        return reg_strength * tf.reduce_sum(from_pt_l2)

    model.add_regularizer(regularizer)

    return model


@executable.executable()
def glue_finetuning_metrics(model, tasks):
    return model.create_metrics()


@executable.executable(
    default_bindings={
        "initializer": bert_initializer,
        "loader": bert_loader,
        "builder": bert_builder,
        "metrics": glue_finetuning_metrics,
    }
)
def bert_finetuning_model(
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


###############################################################################
