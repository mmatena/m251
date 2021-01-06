"""TODO: Add title."""
import abc

from del8.core import data_class
from del8.core.di import scopes
from del8.core.experiment import group
from del8.core.experiment import runs

from del8.core.utils import project_util
from del8.storages.gcp import gcp
from del8.storages.gcp import preloading


@group.group(
    uuid="5c68fddf85804d8cb3fa0a821dc2c336",
    # We use most of the default values for the gcp storage.
    storage_params=gcp.GcpStorageParams(
        preloading_params=preloading.GcpPreloadingParams(
            clear_style=preloading.GcpPreloadingParams.DELETE_UNUSED
        )
    ),
    name="Merging of fine-tuned BERT models on GLUE preliminary experiments.",
    description="",
    project_params=[
        project_util.ProjectParams(folder_path="~/Desktop/projects/del8"),
        project_util.ProjectParams(folder_path="~/Desktop/projects/m251"),
    ],
    extra_pip_packages=[
        "absl-py",
        "bayesian-optimization==1.2.0",
        "bert-for-tf2==0.14.6",
        "google-auth==1.19.2",
        "google-cloud-storage==1.30.0",
        "google-resumable-media==0.7.0",
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
class PaperExpGroup(object):
    pass


###############################################################################


class ExperimentAbc(abc.ABC):
    """Reduces boilerplate."""

    def create_run_instance_config(self, params):
        return runs.RunInstanceConfig(
            global_binding_specs=params.create_binding_specs()
        )

    def create_preload_blob_uuids(self, params):
        return params.create_preload_blob_uuids()


class ParamsAbc(abc.ABC):
    """Reduces boilerplate."""

    @abc.abstractmethod
    def create_bindings(self):
        """Should return Dict[name, binding]."""
        raise NotImplementedError

    def create_binding_specs(self):
        return [
            scopes.ArgNameBindingSpec(name, binding)
            for name, binding in self.create_bindings().items()
        ]

    def create_preload_blob_uuids(self):
        return None
