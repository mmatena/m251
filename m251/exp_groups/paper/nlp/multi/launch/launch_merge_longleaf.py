"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/multi/launch/launch_merge_longleaf.py

"""
from del8.executors.gce import gce
from del8.executors.vastai import vastai
from del8.executors.vastai import api_wrapper

from m251.exp_groups.paper.nlp.multi import merge

from itertools import islice
import uuid as uuidlib

from del8.core import serialization
from del8.core.not_sync import events
from del8.core.utils import project_util

from del8.executors.longleaf import longleaf
from del8.executors.longleaf import longleaf_util
from del8.executors.longleaf import slurm
from del8.executors.longleaf.slurm_interface import slurm_interface
from del8.executors.longleaf.supervisor import supervisor
from del8.executors.longleaf.worker import worker

from del8.storages.gcp import gcp
from del8.storages.gcp import preloading


EXP = merge.Merge_Most

EXP.no_gcs_connect = True
execution_items = EXP.create_all_execution_items()
print(f"Number of execution items to process: {len(execution_items)}")

longleaf_params = longleaf.LongleafParams(
    user="mmatena",
    storage_params=gcp.GcpStorageParams(
        preloading_params=preloading.GcpPreloadingParams(
            clear_style=preloading.GcpPreloadingParams.DELETE_NONE,
            preload_dir="/pine/scr/m/m/mmatena/del8_launches/del8_gcp_preload_dir",
        )
    ),
    project_params=[
        project_util.ProjectParams(folder_path="~/Desktop/projects/del8"),
        project_util.ProjectParams(folder_path="~/Desktop/projects/m251"),
    ],
    supervisor_params=supervisor.SupervisorParams(
        target_num_workers=1,
        num_buffer_workers=1,
        max_buffer_queue_size=2,
        slurm_params=slurm.SlurmParams(
            ram_gb=4,
            partition="general",
            num_cpus=2,
            duration="2:00:00",
        ),
    ),
    worker_params=worker.WorkerParams(
        slurm_params=slurm.SlurmParams(
            ram_gb=32,
            partition="volta-gpu",
            # partition='gpu',
            num_cpus=12,
            duration="2:00:00",
            num_gpus=1,
        ),
        **EXP.get_all_package_kwargs([]),
    ),
    slurm_interface_params=slurm_interface.SlurmInterfaceParams(),
)


longleaf.launch(execution_items, longleaf_params)
