"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/intermediate_hf/launch/launch_eval.py

"""
from del8.executors.gce import gce
from del8.executors.vastai import vastai
from del8.executors.vastai import api_wrapper

from m251.exp_groups.paper.nlp.intermediate_hf import eval
from m251.exp_groups.paper.nlp.intermediate_hf import eval_large


# execution_items = []
# EXP = eval.Eval_LrSrc_Sft_LastCkpt
# execution_items.extend(EXP.create_all_execution_items())
# EXP = eval.Eval_HrSrc_Sft_LastCkpt
# execution_items.extend(EXP.create_all_execution_items())

# EXP = eval.EvalBertBaseFromMnli_Sft_LastCkpt
EXP = eval.EvalGlueFinetune_BertBase_Squad_Sft

# EXP = eval_large.Eval_Large_Sft_LastCkpt

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
                "dph < 2.25",
                "inet_down > 100",
                "inet_up > 75",
                # "gpu_ram >= 10",
                # "dlperf >= 16",
                "cuda_vers >= 11.0 has_avx = true",
            ]
        ),
        order_str="dlperf_usd-",
    ),
    disk_gb=16,
    image="tensorflow/tensorflow:2.4.0-gpu",
)

offers = api_wrapper.query_offers(vast_params)
print(f"Number of acceptable offers: {len(offers)}")

launch_params = gce.GceParams()
node, deploy = gce.launch(execution_items, vast_params, launch_params)
