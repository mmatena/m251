"""Common fisher @executables."""
import tempfile

from absl import logging
import tensorflow as tf

from del8.core import data_class
from del8.core.di import executable
from del8.core.di import scopes


@data_class.data_class()
class SavedFisherMatrix(object):
    def __init__(self, fisher_type, blob_uuid):
        pass


@executable.executable()
def fisher_matrix_saver(fisher_matrix, storage, fisher_type):
    suffix = fisher_matrix.get_file_suffix()
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        fisher_matrix.save(f.name)
        blob_uuid = storage.store_blob_from_file(f.name)

    saved_fisher_matrix = SavedFisherMatrix(
        fisher_type=fisher_type,
        blob_uuid=blob_uuid,
    )
    item_uuid = storage.store_item(saved_fisher_matrix)

    return saved_fisher_matrix, item_uuid


###############################################################################


@executable.executable(
    default_bindings={
        "fisher_matrix_saver": fisher_matrix_saver,
        "split": "train",
    },
)
def fisher_computation(
    dataset, compiled_fisher_computer, _fisher_matrix_saver, num_dataset_passes=1
):
    # The `num_dataset_passes` binding is useful when we are sampling to compute the Fisher.
    compiled_fisher_computer.fit(dataset, epochs=num_dataset_passes)
    fisher_matrix = compiled_fisher_computer.get_fisher_matrix()
    saved_fisher_matrix = _fisher_matrix_saver(fisher_matrix)
    return saved_fisher_matrix


###############################################################################


@data_class.data_class()
class FisherMatricesSummary(object):
    def __init__(self, saved_fisher_matrix_uuids=()):
        pass


class _SaveFisherCallback(tf.keras.callbacks.Callback):
    def __init__(self, fisher_saver):
        super().__init__()
        self.fisher_saver = fisher_saver

    def on_train_begin(self, logs=None):
        self.fisher_saver.reset()

    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Saving fisher matrix at epoch {epoch}.")
        self.fisher_saver.save_fisher(self.model)


@executable.executable()
class epochal_fisher_saver(object):
    # NOTE: This is stateful. I should think about stateful vs stateless executables
    # and see if we should differentiate between the two in the framework (or, far
    # less likely, outright ban stateful ones).

    def __init__(self):
        self.reset()

    def reset(self, storage, _fisher_matrix_saver):
        self.summary = None
        self.summary_uuid = None
        self.storage = storage
        self._fisher_matrix_saver = _fisher_matrix_saver

    def save_fisher(self, model):
        fisher_matrix = model.get_fisher_matrix()
        _, item_uuid = self._fisher_matrix_saver(fisher_matrix)
        self.add_fisher_to_summary(item_uuid)

    def add_fisher_to_summary(self, item_uuid):
        if self.summary_uuid is None:
            self.initialize_summary()

        self.summary = self.summary.copy(
            saved_fisher_matrix_uuids=self.summary.saved_fisher_matrix_uuids
            + (item_uuid,)
        )

        self.storage.replace_item(self.summary_uuid, self.summary)

    def initialize_summary(self):
        if self.summary_uuid is not None:
            raise ValueError(
                "Tried to create a new epochal fisher matric summary but one is already "
                "associated with the epochal fisher matric saver."
            )

        self.summary = FisherMatricesSummary()
        self.summary_uuid = self.storage.store_item(self.summary)

    def call(self, storage):
        return _SaveFisherCallback(self)


@executable.executable(
    default_bindings={
        "fisher_matrix_saver": fisher_matrix_saver,
        "save_epochal_fisher_callback": epochal_fisher_saver,
        "split": "train",
        "repeat": True,
        "shuffle": True,
    },
)
def variational_fisher_computation(
    dataset,
    compiled_fisher_computer,
    _fisher_matrix_saver,
    epochs,
    steps_per_epoch,
    save_epochal_fisher_callback,
    save_fisher_at_each_epoch=False,
):
    callbacks = []
    if save_fisher_at_each_epoch:
        callbacks.append(save_epochal_fisher_callback)

    compiled_fisher_computer.fit(
        dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks
    )

    if not save_fisher_at_each_epoch:
        fisher_matrix = compiled_fisher_computer.get_fisher_matrix()
        saved_fisher_matrix, item_uuid = _fisher_matrix_saver(fisher_matrix)
        return saved_fisher_matrix, item_uuid
    else:
        return None, None
