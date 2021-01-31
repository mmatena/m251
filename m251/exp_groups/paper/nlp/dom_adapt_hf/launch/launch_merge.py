"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/dom_adapt_hf/launch/launch_merge.py

"""
from del8.executors.gce import gce
from del8.executors.vastai import vastai
from del8.executors.vastai import api_wrapper

from m251.exp_groups.paper.nlp.dom_adapt_hf import merge
from m251.exp_groups.paper.nlp.dom_adapt_hf import merge2
from m251.exp_groups.paper.nlp.dom_adapt_hf import merge3
from m251.exp_groups.paper.nlp.dom_adapt_hf import merge4
from m251.exp_groups.paper.nlp.dom_adapt_hf import merge5


# EXP = merge.Merge_MlmS2orc_Normalized
# EXP = merge.Merge_MlmS2orc_Normalized_Mlm4096
# EXP = merge.Merge_MlmS2orc_Normalized_131072_FOR_REAL
# EXP = merge2.Merge_MlmS2orc_ROBERTA
# EXP = merge2.Merge_ROBERTA_LastCkpt_TestSet
# EXP = merge2.Merge_ROBERTA_LastCkpt_WrongMerge_TestSet
# EXP = merge3.Merge_ROBERTA_LastCkpt_TestSet
# EXP = merge3.Merge_ROBERTA_LastCkpt_TestSet_Clipped1
# EXP = merge4.Merge_ROBERTA_LastCkpt_TestSet_WithDaptAllVars
# EXP = merge4.Merge_ROBERTA_LastCkpt_TestSet_WithDaptAllVars_MergeOnlyBody
# EXP = merge3.Merge_ROBERTA_LastCkpt_TestSet_131072
# EXP = merge5.Merge_ROBERTA_LastCkpt_TestSet_PretrainedMore
# EXP = merge5.Merge_ROBERTA_LastCkpt_TestSet_PretrainedMore_REAL
# EXP = merge5.Merge_ROBERTA_LastCkpt_TestSet_PretrainFromDapt32768
EXP = merge5.Merge_ROBERTA_LastCkpt_TestSet_Pretrain32768NoReg


execution_items = EXP.create_all_execution_items()
print(f"Number of execution items to process: {len(execution_items)}")

vast_params = vastai.create_supervisor_params(
    EXP,
    execution_items=execution_items,
    num_workers=8,
    offer_query=vastai.OfferQuery(
        queries_str="  ".join(
            [
                "reliability > 0.95",
                "num_gpus=1",
                "dph < 0.5",
                "inet_down > 50",
                "inet_up > 50",
                "gpu_ram >= 10",
                # "dlperf >= 16",
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

# execution_items = [
#     item for item in execution_items
#     if item.worker_run_kwargs['run_params'].models_to_merge[0].task == 'sci_erc'
#     # if item.worker_run_kwargs['run_params'].models_to_merge[0].task == 'chemprot'
#     # if item.worker_run_kwargs['run_params'].models_to_merge[0].task == 'acl_arc'
# ]

# entrypoint.worker_run(**execution_items[0].worker_run_kwargs)
# # entrypoint.worker_run(**execution_items[-1].worker_run_kwargs)

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
