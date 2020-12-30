"""TODO: Add title."""
from del8.core import data_class
from del8.core.experiment import group

from del8.core.utils import project_util
from del8.storages.gcp import gcp


@group.group(
    uuid="6eebd2c79d6248eb8c61fc7419c6dc45",
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
        "h5py",
        "numpy",
        "params-flow",
        "pinject",
        "psycopg2-binary",
        "requests",
        "scikit-learn==0.23.1",
        "scipy==1.4.1",
        "sshtunnel",
        "tensorflow-datasets==4.1.0",
        "tensorflow-probability==0.11.1",
        "transformers==3.0.2",
    ],
)
class BertMergingPrelimsGroup(object):
    pass
