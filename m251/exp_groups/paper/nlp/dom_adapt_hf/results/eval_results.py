"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/dom_adapt_hf/results/eval_results.py

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
        print(res.results)
        items.append(
            {
                "task": params.task,
                "trial_index": params.trial_index,
                # "score": res.results,
                "score": {"f1": res.results["f1"]},
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

        #
        #
        metric = "f1"
        #
        #

        res = eval_run.get_items_by_class(eval_execs.CheckpointEvaluationResults)
        print(res[-1].results)
        best = max(res, key=lambda r: get_single_score(r.results[params.task][metric]))
        # print(res.index(best))
        items.append(
            {
                "task": params.task,
                "trial_index": params.trial_index,
                #
                #
                # "score": best.results,
                "score": {params.task: {metric: best.results[params.task][metric]}},
                #
                #
                # "mlm_examples": EVAL_RUN_UUID_TO_MLM_EXAMPLES[params.run_uuid],
                "mlm_examples": "-",
                #
                #
            }
        )

    return items


def create_csv_table(items, round_digits=1):

    row_groups = collections.defaultdict(list)
    for item in items:
        group_key = (item["task"], item["mlm_examples"])
        row_groups[group_key].append(item)

    header = [
        "task",
        "mlm ex",
        "task f1",
        "stddev",
        "num trials",
    ]
    body = []
    for (task, mlm_examples), row_items in row_groups.items():
        scores = np.array([get_single_score(item["score"]) for item in row_items])
        row = [
            task,
            mlm_examples,
            round(np.mean(scores), round_digits),
            round(np.std(scores), round_digits),
            len(row_items),
        ]
        body.append(row)

    body = sorted(body, key=lambda r: r[:1])

    rows = [header] + body

    return result_utils.csv_to_str(rows)


EVAL_RUN_UUID_TO_MLM_EXAMPLES = {
    "bf7f40cba213434ca8eccab5779ab6f4": 262144,
    "720349e5bec348e8943eefb863800eb4": 262144,
    "3949a9956b1344b4bc35fd2af0f7e92a": 262144,
    "e0784b1e28214099a6e754e7f56f034e": 262144,
    "ff657abaf07148a3b95fc8e90709cdae": 262144,
    "ff18e34f354944bf95893279b9381258": 262144,
    "166aa6303cb0422b9e4d2c2970a04577": 262144,
    "c47133731740473685201a8a848c19ea": 262144,
    "5d774cc90a784694a6b6a7d34b5129bb": 262144,
    "0872fc692949493f9ffbffe01497ca44": 262144,
    "59a5128c22134474bfa8fb0544397394": 2097152,
    "5f7fe8370649484a977ace31fafb9443": 2097152,
    "61989a8eea034bc1a4cad432b5f8e901": 2097152,
    "762b244f982f4b93b6467e6b72be32a5": 2097152,
    "c55f65653efe432597168ae9c30f7e16": 2097152,
    "867357f5ddae41c1897dda6ddbe64734": 2097152,
    "16ab8138adce49aaaa7deb3b18f514e9": 2097152,
    "cd3ccedaebf5475b8a6fe76ec8c7c2fd": 2097152,
    "3b98bf3b229845c5a39e59a037e5525b": 2097152,
}


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.dom_adapt_hf import eval as evl
    from m251.exp_groups.paper.nlp.dom_adapt_hf import eval2 as evl2
    from m251.exp_groups.paper.nlp.dom_adapt_hf import eval_test_set
    from m251.exp_groups.paper.nlp.dom_adapt_hf import eval_test_set2

    ###########################################################################

    # merge_exp = evl.Eval_LowResource_Dapt_All
    # merge_exp = evl.Finetune_Dapt_LowResource_All_FOR_REAL
    # merge_exp = eval_test_set.Finetune_Dapt_LowResource_All_FOR_REAL_Best_Original
    # merge_exp = eval_test_set.Eval_LowResource_All_FOR_REAL_Best_Original
    # merge_exp = evl2.Eval_DAPT_LowResource_All
    # merge_exp = eval_test_set2.EvalTest_Original_DAPT_Best
    # merge_exp = evl2.Eval_Finetune_Pretrained32768_All_TestSet
    # merge_exp = evl2.Eval_Finetune_PretrainMore_All_TestSet
    merge_exp = evl2.Eval_FinetunePretrain32768NoReg_All_TestSet

    summary = create_json_best(merge_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)

    t = create_csv_table(summary)
    print(t)
