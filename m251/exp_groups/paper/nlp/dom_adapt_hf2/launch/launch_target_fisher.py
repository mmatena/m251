"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/dom_adapt_hf2/launch/launch_target_fisher.py

"""
from del8.executors.gce import gce
from del8.executors.vastai import vastai
from del8.executors.vastai import api_wrapper

from m251.exp_groups.paper.nlp.dom_adapt_hf2 import target_fisher


# EXP = target_fisher.Fisher_Cs_32768_1e6
# EXP = target_fisher.Fisher_DAPT_CsFt_AllCkpts
EXP = target_fisher.Fisher_DAPT_BioMedFt_AllCkpts


execution_items = EXP.create_all_execution_items()
print(f"Number of execution items to process: {len(execution_items)}")

vast_params = vastai.create_supervisor_params(
    EXP,
    execution_items=execution_items,
    num_workers=10,
    offer_query=vastai.OfferQuery(
        queries_str="  ".join(
            [
                "reliability > 0.95",
                "num_gpus=1",
                "dph < 2.0",
                "inet_down > 50",
                "inet_up > 50",
                # "gpu_ram >= 20",
                # "dlperf >= 16",
                "cuda_vers >= 11.0 has_avx = true",
                "cuda_vers <= 11.1",
            ]
        ),
        order_str="dlperf_usd-",
    ),
    disk_gb=20,
    image="tensorflow/tensorflow:2.4.0-gpu",
)

offers = api_wrapper.query_offers(vast_params)
print(f"Number of acceptable offers: {len(offers)}")

launch_params = gce.GceParams()
node, deploy = gce.launch(execution_items, vast_params, launch_params)

# #
# #
# #
# ###############################################################################
# # Stages of testing:
# ###############################################################################
# if True:
#     from del8.core.execution import entrypoint
#     from del8.storages.gcp import gcp

# gcp.PERSISTENT_CACHE = True
# EXP.to_dev_mode()

# execution_items = EXP.create_all_execution_items(skip_finished=False)
# print(f'Number of execution items to process: {len(execution_items)}')

# entrypoint.worker_run(**execution_items[0].worker_run_kwargs)

# ###############################################################################
