"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/ablations/reg_intermediate/launch/launch_fisher.py

"""
from del8.executors.gce import gce
from del8.executors.vastai import vastai
from del8.executors.vastai import api_wrapper

from m251.exp_groups.paper.ablations.reg_intermediate import fisher


EXP = fisher.FisherComputation

launch_params = gce.GceParams()

vast_params = vastai.create_supervisor_params(
    EXP,
    num_workers=6,
    offer_query=vastai.OfferQuery(
        queries_str="  ".join(
            [
                "reliability > 0.95",
                "num_gpus=1",
                "dph < 0.5",
                "inet_down > 75",
                "inet_up > 75",
                "dlperf >= 16",
                "cuda_vers >= 11.0 has_avx = true",
            ]
        ),
        order_str="dlperf_usd-",
    ),
    disk_gb=12,
)

offers = api_wrapper.query_offers(vast_params)
print(f"Number of acceptable offers: {len(offers)}")

execution_items = EXP.create_all_execution_items()
print(f"Number of execution items to process: {len(execution_items)}")

node, deploy = gce.launch(execution_items, vast_params, launch_params)
