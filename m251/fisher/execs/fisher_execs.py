"""Common fisher @executables."""
import tempfile

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
    storage.store_item(saved_fisher_matrix)

    return saved_fisher_matrix


@executable.executable(
    default_bindings={
        "fisher_matrix_saver": fisher_matrix_saver,
        "split": "train",
    },
)
def fisher_computation(dataset, compiled_fisher_computer, _fisher_matrix_saver):
    compiled_fisher_computer.fit(dataset, epochs=1)
    fisher_matrix = compiled_fisher_computer.get_fisher_matrix()
    saved_fisher_matrix = _fisher_matrix_saver(fisher_matrix)
    return saved_fisher_matrix
