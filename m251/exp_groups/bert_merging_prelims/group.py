"""TODO: Add title."""
from del8.core import data_class
from del8.core.experiment import group

from del8.core.utils import project_util
from del8.storages.gcp import gcp


@group.group(
    uuid="a982ac2a4f964f879e514ae2a8d4b1eb",
    # We use all of the default values for the gcp storage.
    storage_params=gcp.GcpStorageParams(),
    name="Merging of fine-tuned BERT models on GLUE preliminary experiments.",
    description="",
    project_params=[
        # NOTE: Maybe add extra ignores for local_scripts
        project_util.ProjectParams(folder_path="~/Desktop/projects/del8"),
        project_util.ProjectParams(folder_path="~/Desktop/projects/m251"),
    ],
    extra_pip_packages=[
        # TODO: Figure our whether this or auto-injection?
        "absl-py",
        "bert-for-tf2",
        "google-auth",
        "google-cloud-storage",
        "params-flow",
        "pinject",
        "psycopg2-binary",
        "requests",
        "sshtunnel",
        "tensorflow-datasets",
        "tensorflow-probability",
        "transformers",
    ],
)
class BertMergingPrelimsGroup(object):
    pass
