"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/dom_adapt/results/eval_results.py

"""
import collections
import csv
import json
import os

import numpy as np

from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.evaluation import eval_execs

from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES
from m251.fisher.execs import merging_execs
from m251.exp_groups.paper.results import utils as result_utils

get_single_score = result_utils.get_single_score
result_file = result_utils.result_file


def create_json(eval_exp):
    with eval_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[eval_exp.uuid])

    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=eval_exp.uuid)

    items = []
    for run_id in run_ids:
        eval_run = exps_data.get_run_data(run_id)

        params = eval_run.get_single_item_by_class(eval_exp.params_cls)
        res = eval_run.get_single_item_by_class(eval_execs.CheckpointEvaluationResults)

        items.append(
            {
                "task": params.task,
                "trial_index": params.trial_index,
                "score": res.results,
            }
        )

    return items


def create_json_best(eval_exp):
    with eval_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[eval_exp.uuid])

    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=eval_exp.uuid)

    items = []
    for run_id in run_ids:
        eval_run = exps_data.get_run_data(run_id)

        params = eval_run.get_single_item_by_class(eval_exp.params_cls)
        res = eval_run.get_items_by_class(eval_execs.CheckpointEvaluationResults)
        best = max(res, key=lambda r: get_single_score(r.results))
        items.append(
            {
                "task": params.task,
                "trial_index": params.trial_index,
                "score": best.results,
            }
        )

    return items


# def create_json_all(train_exp, eval_exp):
#     with eval_exp.get_storage() as storage:
#         exps_data = storage.retrieve_storage_data(
#             experiment_uuid=[eval_exp.uuid, train_exp.uuid])

#     run_ids = exps_data.get_finished_runs_ids(experiment_uuid=eval_exp.uuid)

#     items = []
#     for run_id in run_ids:
#         eval_run = exps_data.get_run_data(run_id)

#         params = eval_run.get_single_item_by_class(eval_exp.params_cls)
#         res = eval_run.get_items_by_class(eval_execs.CheckpointEvaluationResults)
#         best = max(res, key=lambda r: get_single_score(r.results))
#         items.append(
#             {
#                 "task": params.task,
#                 "trial_index": params.trial_index,
#                 "score": best.results,
#             }
#         )

#     return items


def create_csv_table(items, round_digits=1):

    row_groups = collections.defaultdict(list)
    for item in items:
        group_key = item["task"]
        row_groups[group_key].append(item)

    header = [
        "task",
        "task f1",
        "stddev",
        "num trials",
    ]
    body = []
    for task, row_items in row_groups.items():
        scores = np.array([get_single_score(item["score"]) for item in row_items])
        row = [
            task,
            round(np.mean(scores), round_digits),
            round(np.std(scores), round_digits),
            len(row_items),
        ]
        body.append(row)

    body = sorted(body, key=lambda r: r[:1])

    rows = [header] + body

    return result_utils.csv_to_str(rows)


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.dom_adapt import eval as evl

    ###########################################################################

    merge_exp = evl.Eval_LowResource_NoDapt_All

    summary = create_json_best(merge_exp)
    s = json.dumps(summary, indent=2)
    print(s)

    t = create_csv_table(summary)
    print(t)

    ###########################################################################

    # merge_exp = evl.Eval_LowResource_Dapt_All

    # summary = create_json_best(merge_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)

    # t = create_csv_table(summary)
    # print(t)

    ###########################################################################

    # merge_exp = evl.Eval_LowResource_Dapt

    # summary = create_json(merge_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)

    # t = create_csv_table(summary)
    # print(t)
