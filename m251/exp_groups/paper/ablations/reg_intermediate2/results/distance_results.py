"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/ablations/reg_intermediate2/results/distance_results.py

"""
import collections
import csv
import json
import os

import numpy as np

from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES
from m251.fisher.execs import merging_execs
from m251.exp_groups.paper.results import utils as result_utils
from m251.exp_groups.paper.ablations.reg_intermediate2 import distance_callback

get_single_score = result_utils.get_single_score
result_file = result_utils.result_file


# TEMP_JSON = "/tmp/merge_temp.json"
JSON_FILE = result_file("distance_results.json")


def create_json(exps):
    with exps[0].get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[e.uuid for e in exps]
        )
    items = []

    for exp in exps:
        run_ids = exps_data.get_finished_runs_ids(experiment_uuid=exp.uuid)

        for run_id in run_ids:
            merge_run = exps_data.get_run_data(run_id)

            params = merge_run.get_single_item_by_class(exp.params_cls)
            summary = merge_run.get_single_item_by_class(
                distance_callback.DistanceSummary
            )
            print(params.reg_strength)
            items.append(
                {
                    "task": params.task,
                    "reg_strength": params.reg_strength,
                    "trial_index": params.trial_index,
                    "sq_l2_per_step": summary.sq_l2_per_step,
                }
            )

    return items


if __name__ == "__main__":
    from m251.exp_groups.paper.ablations.reg_intermediate2 import record_distances

    ###########################################################################

    exps = [record_distances.Finetune_Rte, record_distances.Finetune_Mnli]
    summary = create_json(exps)
    # s = json.dumps(summary, indent=2)
    # print(s)
    with open(JSON_FILE, "w") as f:
        json.dump(summary, f, indent=2)
