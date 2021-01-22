"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/dom_adapt/results/merge_results.py

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


MERGE_MLM_TARGET_TASK_JSON = result_file("nlp/dom_adapt/merge_mlm_target_task.json")

MERGE_MLM_TARGET_TASK_NORMALIZED_LAST_CKPT_JSON = result_file(
    "nlp/dom_adapt/merge_mlm_target_task_normalized_last_ckpt.json"
)

MERGE_MLM_TARGET_TASK_NORMALIZED_ACL_ARC_BEST_CKPT_JSON = result_file(
    "nlp/dom_adapt/merge_mlm_target_task_normalized_acl_arc_best_ckpt.json"
)

MERGE_MLM_TARGET_TASK_NORMALIZED_SCI_ERC_CKPT_8_JSON = result_file(
    "nlp/dom_adapt/merge_mlm_target_task_normalized_sci_erc_ckpt_8.json"
)

MERGE_MLM_S2ORC_NORMALIZED_LAST_CKPT_JSON = result_file(
    "nlp/dom_adapt/merge_mlm_s2orc_normalized_last_ckpt.json"
)

MERGE_MLM_S2ORC_NORMALIZED_BEST_CKPT_JSON = result_file(
    "nlp/dom_adapt/merge_mlm_s2orc_normalized_best_ckpt.json"
)

MERGE_MLM_S2ORC_NORMALIZED_ALL_CKPT_JSON = result_file(
    "nlp/dom_adapt/merge_mlm_s2orc_normalized_all_ckpt.json"
)


def create_json(merge_exp, group_by_ckpt_index=True, group_by_finetuned_model=False):
    with merge_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[merge_exp.uuid])

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=merge_exp.uuid)

    items = []
    for run_id in merge_run_ids:
        merge_run = exps_data.get_run_data(run_id)

        params = merge_run.get_single_item_by_class(merge_exp.params_cls)
        reses = merge_run.get_items_by_class(merging_execs.MergingEvaluationResults)

        # print([(r.weighting[0], get_single_score(r.results)) for r in reses])

        res = max(reses, key=lambda r: get_single_score(r.results))
        og_res = max(reses, key=lambda r: r.weighting[0])
        donor_body_res = max(reses, key=lambda r: r.weighting[1])

        assert og_res.weighting[0] == 1.0
        assert donor_body_res.weighting[1] == 1.0

        target_mtm, donor_mtm = params.models_to_merge
        hyperparams = {
            "task": target_mtm.task,
        }
        if group_by_ckpt_index:
            hyperparams["target_ckpt_index"] = params.target_ckpt_index
        if group_by_finetuned_model:
            hyperparams["train_run_uuid"] = target_mtm.train_run_uuid
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


def create_csv_table(
    filepath, round_digits=1, group_by_ckpt_index=True, best_per_finetuned_model=False
):
    items = result_utils.load_json(filepath)

    row_groups = collections.defaultdict(list)
    for item in items:
        group_key = hashabledict(item["hyperparams"])
        row_groups[group_key].append(item)

    if best_per_finetuned_model:
        new_row_groups = collections.defaultdict(list)
        for hp, row_items in row_groups.items():
            # TODO: get best original score as well
            best = max(row_items, key=lambda r: get_single_score(r["merged_score"]))

            new_key = dict(hp)
            del new_key["train_run_uuid"]
            new_key = hashabledict(new_key)
            new_row_groups[new_key].append(best)
        row_groups = new_row_groups

    header = [
        "task",
        "task ckpt",
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
    for hp, row_items in row_groups.items():
        og_scores = np.array(
            [get_single_score(item["original_score"]) for item in row_items]
        )
        merged_scores = np.array(
            [get_single_score(item["merged_score"]) for item in row_items]
        )
        row = [
            hp["task"],
            hp["target_ckpt_index"] if group_by_ckpt_index else "-",
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

    body = sorted(body, key=lambda r: r[:2])

    rows = [header] + body

    return result_utils.csv_to_str(rows)


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.dom_adapt import fisher
    from m251.exp_groups.paper.nlp.dom_adapt import merge

    ###########################################################################

    merge_exp = merge.Merge_MlmS2orc_Normalized_AllCkpt
    summary = create_json(
        merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    )
    filepath = MERGE_MLM_S2ORC_NORMALIZED_ALL_CKPT_JSON
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    t = create_csv_table(
        filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    )
    print(t)

    ###########################################################################

    # merge_exp = merge.Merge_MlmS2orc_Normalized_BestCkpt
    # summary = create_json(merge_exp, group_by_ckpt_index=False)
    # filepath = MERGE_MLM_S2ORC_NORMALIZED_BEST_CKPT_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath, group_by_ckpt_index=False)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_MlmS2orc_Normalized_LastCkpt
    # summary = create_json(merge_exp)
    # filepath = MERGE_MLM_S2ORC_NORMALIZED_LAST_CKPT_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_MlmTargetTask_Normalized_SciErc_Ckpt8
    # summary = create_json(merge_exp)
    # filepath = MERGE_MLM_TARGET_TASK_NORMALIZED_SCI_ERC_CKPT_8_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_MlmTargetTask_Normalized_AclArc_BestCkpt
    # summary = create_json(merge_exp)
    # filepath = MERGE_MLM_TARGET_TASK_NORMALIZED_LAST_CKPT_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # filepath = MERGE_MLM_TARGET_TASK_NORMALIZED_LAST_CKPT_JSON
    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_MlmTargetTask_Normalized_LastCkpt
    # summary = create_json(merge_exp)
    # with open(MERGE_MLM_TARGET_TASK_NORMALIZED_LAST_CKPT_JSON, "w") as f:
    #     json.dump(summary, f, indent=2)

    ###########################################################################

    # filepath = MERGE_MLM_TARGET_TASK_JSON
    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_MlmTargetTask
    # summary = create_json(merge_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)
    # with open(MERGE_MLM_TARGET_TASK_JSON, "w") as f:
    #     json.dump(summary, f, indent=2)
