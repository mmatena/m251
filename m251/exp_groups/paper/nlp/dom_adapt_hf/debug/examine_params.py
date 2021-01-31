"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/dom_adapt_hf/debug/examine_params.py

"""
import collections
import csv
import json
import os

import numpy as np

from del8.core import serialization
from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.evaluation import eval_execs

from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES
from m251.fisher.execs import merging_execs
from m251.exp_groups.paper.results import utils as result_utils

get_single_score = result_utils.get_single_score
result_file = result_utils.result_file


def examine_params(eval_exp):
    with eval_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[eval_exp.uuid])

    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=eval_exp.uuid)

    for run_id in run_ids:
        eval_run = exps_data.get_run_data(run_id)

        params = eval_run.get_single_item_by_class(eval_exp.params_cls)
        jason = serialization.to_pretty_json(params)
        print(json.dumps(jason, indent=2))


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.dom_adapt_hf import eval as evl
    from m251.exp_groups.paper.nlp.dom_adapt_hf import eval_test_set
    from m251.exp_groups.paper.nlp.dom_adapt_hf import finetune

    ###########################################################################

    # exp = finetune.Finetune_Dapt_LowResource_FOR_REAL
    exp = finetune.Finetune_LowResource
    examine_params(exp)

    ###########################################################################

    # DAPT_EVAL_EXPS = [
    #     evl.Finetune_Dapt_LowResource_All_FOR_REAL,
    #     eval_test_set.Finetune_Dapt_LowResource_All_FOR_REAL_Best_Original,
    # ]

    # for exp in DAPT_EVAL_EXPS:
    #     print(exp.__class__.__name__)
    #     examine_params(exp)
    #     print('\n')
