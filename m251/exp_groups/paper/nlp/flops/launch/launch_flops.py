"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/flops/launch/launch_flops.py

"""
from del8.executors.gce import gce
from del8.executors.vastai import vastai
from del8.executors.vastai import api_wrapper

from m251.exp_groups.paper.nlp.flops import eval_flops


EXP = eval_flops.Finetune_Rte

#
#
#
###############################################################################
# Stages of testing:
###############################################################################
if True:
    from del8.core.execution import entrypoint
    from del8.storages.gcp import gcp

gcp.PERSISTENT_CACHE = True
EXP.to_dev_mode()

execution_items = EXP.create_all_execution_items(skip_finished=False)
print(f"Number of execution items to process: {len(execution_items)}")

entrypoint.worker_run(**execution_items[0].worker_run_kwargs)
