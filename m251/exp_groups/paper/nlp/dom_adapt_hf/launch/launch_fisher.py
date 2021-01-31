"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/dom_adapt_hf/launch/launch_fisher.py

"""
from del8.executors.gce import gce
from del8.executors.vastai import vastai
from del8.executors.vastai import api_wrapper

from m251.exp_groups.paper.nlp.dom_adapt_hf import fisher
from m251.exp_groups.paper.nlp.dom_adapt_hf import fisher2
from m251.exp_groups.paper.nlp.dom_adapt_hf import fisher3


# EXP = fisher.FisherComputation_Dapt_TargetTask_FOR_REAL
# EXP = fisher.FisherComputation_TargetTask_FOR_REAL
# EXP = fisher2.FisherComputation_ROBERTA_TargetTasks
# EXP = fisher2.FisherComputation_MlmS2orc_16384
# EXP = fisher2.FisherComputation_MlmS2orc_16384_Clipped1
# EXP = fisher2.FisherComputation_ROBERTA_TargetTasks_LastCkpt_AllVars
# EXP = fisher2.FisherComputation_DAPT_HeadOnly_TargetTasks_LastCkpt_AllVars
# EXP = fisher2.FisherComputation_MlmS2orc_131072
# EXP = fisher2.FisherComputation_MlmS2orc_1048576
# EXP = fisher3.Fisher_PretrainMore
# EXP = fisher3.Fisher_PretrainMore_REAL
# EXP = fisher3.Fisher_PretrainFromDapt32768
EXP = fisher3.Fisher_Pretrain32768NoReg


execution_items = EXP.create_all_execution_items()
print(f"Number of execution items to process: {len(execution_items)}")

vast_params = vastai.create_supervisor_params(
    EXP,
    execution_items=execution_items,
    num_workers=1,
    offer_query=vastai.OfferQuery(
        queries_str="  ".join(
            [
                "reliability > 0.95",
                "num_gpus=1",
                "dph < 0.5",
                "inet_down > 200",
                "inet_up > 50",
                "gpu_ram >= 10",
                "dlperf >= 16",
                "cuda_vers >= 11.0 has_avx = true",
            ]
        ),
        order_str="dlperf_usd-",
    ),
    disk_gb=24,
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

###############################################################################

# EXP.to_dev_mode()

# vastai.launch_experiment(
#     EXP,
#     num_workers=1,
#     offer_query=vastai.OfferQuery(
#         queries_str="  ".join(
#             [
#                 "reliability > 0.95",
#                 "num_gpus=1",
#                 "dph < 0.5",
#                 "inet_down > 200",
#                 "inet_up > 75",
#                 "gpu_ram >= 10",
#                 "dlperf >= 16",
#                 "cuda_vers >= 11.0 has_avx = true",
#             ]
#         ),
#         order_str="dlperf_usd-",
#     ),
#     disk_gb=20,
# )
