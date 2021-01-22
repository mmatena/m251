"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/vision/intermediate/launch/launch_merge.py

"""
from del8.executors.gce import gce
from del8.executors.vastai import vastai
from del8.executors.vastai import api_wrapper

from m251.exp_groups.paper.vision.intermediate import merge
from m251.exp_groups.paper.vision.intermediate import merge_search


# EXP = merge.Merge_1x
# EXP = merge.Merge_1x_Exact
# EXP = merge_search.Merge_1x_Exact
EXP = merge.Merge_4x_DtdTarget
# EXP = merge_search.Merge_4x

execution_items = EXP.create_all_execution_items()
print(f"Number of execution items to process: {len(execution_items)}")

vast_params = vastai.create_supervisor_params(
    EXP,
    execution_items=execution_items,
    num_workers=5,
    offer_query=vastai.OfferQuery(
        queries_str="  ".join(
            [
                "reliability > 0.95",
                "num_gpus=1",
                "dph < 0.5",
                "inet_down > 200",
                "inet_up > 40",
                "gpu_ram >= 10",
                "dlperf >= 16",
                "cuda_vers >= 11.0 has_avx = true",
            ]
        ),
        order_str="dlperf_usd-",
    ),
    disk_gb=32,
)

offers = api_wrapper.query_offers(vast_params)
print(f"Number of acceptable offers: {len(offers)}")

launch_params = gce.GceParams()
node, deploy = gce.launch(execution_items, vast_params, launch_params)


# # #
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

# # entrypoint.worker_run(**execution_items[0].worker_run_kwargs)
# # entrypoint.worker_run(**execution_items[1].worker_run_kwargs)


# def f(params, task1, task2):
#     mtm1, mtm2 = params.models_to_merge
#     # print(mtm1.task, mtm2.task)
#     return mtm1.task == task1 and mtm2.task == task2


# execution_items = [
#     item for item in execution_items
#     if f(item.worker_run_kwargs['run_params'], "dtd", "cifar10")]

# entrypoint.worker_run(**execution_items[0].worker_run_kwargs)

# ###############################################################################

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
