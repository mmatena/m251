"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/dom_adapt_hf2/results/merge_results.py

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

get_single_score = result_utils.get_single_score
result_file = result_utils.result_file


TEMP_JSON = "/tmp/merge_temp.json"


def create_json(merge_exp):
    with merge_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[merge_exp.uuid])

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=merge_exp.uuid)

    items = []
    for run_id in merge_run_ids:
        merge_run = exps_data.get_run_data(run_id)

        try:
            params = merge_run.get_single_item_by_class(merge_exp.params_cls)
        except AssertionError:
            print("Skipping merge run as two params found. Debug this.", merge_run)
            continue
        reses = merge_run.get_items_by_class(merging_execs.MergingEvaluationResults)

        # print(params.target_ckpt_index)

        # print([(r.weighting[0], get_single_score(r.results)) for r in reses])

        res = max(reses, key=lambda r: get_single_score(r.results))
        og_res = max(reses, key=lambda r: r.weighting[0])
        donor_body_res = max(reses, key=lambda r: r.weighting[1])

        # assert og_res.weighting[0] == 1.0
        # assert donor_body_res.weighting[1] == 1.0

        target_mtm, donor_mtm = params.models_to_merge
        hyperparams = {
            "task": target_mtm.task,
            "pretrained_examples": params.pretrained_examples,
            "pretrained_reg_strength": params.pretrained_reg_strength,
            "train_run_uuid": params.models_to_merge[0].train_run_uuid,
        }

        hyperparams["donor_fisher"] = donor_mtm.fisher_run_uuid
        items.append(
            {
                "task": target_mtm.task,
                "other_task": donor_mtm.task,
                "trial_index": params.trial_index,
                "hyperparams": hyperparams,
                "original_score": og_res.results,
                "merged_score": res.results,
                "donor_body_score": donor_body_res.results,
                "weighting": res.weighting[0],
            }
        )

    return items


def create_csv_table(filepath, round_digits=1):
    items = result_utils.load_json(filepath)

    row_groups = collections.defaultdict(list)
    for item in items:
        group_key = hashabledict(item["hyperparams"])
        row_groups[group_key].append(item)

    row_groups2 = collections.defaultdict(list)
    for hp, row_items in row_groups.items():
        best_og = max(
            row_items, key=lambda item: get_single_score(item["original_score"])
        )
        best_merged = max(
            row_items, key=lambda item: get_single_score(item["merged_score"])
        )
        hp = dict(hp)
        del hp["train_run_uuid"]
        group_key = hashabledict(hp)
        row_groups2[group_key].append(
            {
                "original_score": best_og["original_score"],
                "merged_score": best_merged["merged_score"],
            }
        )

    header = [
        "task",
        "mlm train ex",
        "mlm reg str",
        "merged task f1",
        "stddev",
        "orig task f1",
        "stddev",
        "mean boost",
        "stddev",
        "max boost",
        "min boost",
        "num trials",
    ]
    body = []
    for hp, row_items in row_groups2.items():
        og_scores = np.array(
            [get_single_score(item["original_score"]) for item in row_items]
        )
        merged_scores = np.array(
            [get_single_score(item["merged_score"]) for item in row_items]
        )
        row = [
            hp["task"],
            hp["pretrained_examples"],
            hp["pretrained_reg_strength"],
            # q,
            round(np.mean(merged_scores), round_digits),
            round(np.std(merged_scores), round_digits),
            #
            round(np.mean(og_scores), round_digits),
            round(np.std(og_scores), round_digits),
            #
            round(np.mean(merged_scores - og_scores), round_digits),
            round(np.std(merged_scores - og_scores), round_digits),
            #
            round(np.max(merged_scores - og_scores), round_digits),
            round(np.min(merged_scores - og_scores), round_digits),
            len(row_items),
        ]
        body.append(row)

    body = sorted(body, key=lambda r: r[:3])

    rows = [header] + body

    return result_utils.csv_to_str(rows)


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.dom_adapt_hf2 import merge

    ###########################################################################

    # merge_exp = merge.Merge_ROBERTA_LastCkpt_TestSet_PretrainCs
    # merge_exp = merge.Merge_ROBERTA_LastCkpt_TestSet_PretrainBioMed
    # merge_exp = merge.Merge_FinetunedCs327681e6_LastCkpt_TestSet_DAPT131072
    # merge_exp = merge.Merge_ROBERTA_AllCkpts_TestSet_PretrainCs
    merge_exp = merge.Merge_ROBERTA_AllCkpts_TestSet_PretrainBioMed
    # merge_exp = merge.Merge_DAPT_AllCkpts_TestSet_PretrainCs
    # merge_exp = merge.Merge_DAPT_AllCkpts_TestSet_PretrainBioMed

    # merge_exp = merge.DummyMerge_ROBERTA_AllCkpts_TestSet_PretrainCs
    # merge_exp = merge.DummyMerge_ROBERTA_AllCkpts_TestSet_PretrainBioMed
    # merge_exp = merge.DummyMerge_DAPT_AllCkpts_TestSet_PretrainCs
    # merge_exp = merge.DummyMerge_DAPT_AllCkpts_TestSet_PretrainBioMed

    summary = create_json(merge_exp)
    filepath = TEMP_JSON
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    t = create_csv_table(filepath)
    print(t)
