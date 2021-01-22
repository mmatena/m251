"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/intermediate_hf/results/merge_results.py

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

from m251.exp_groups.paper.nlp.intermediate_hf import defs

get_single_score = result_utils.get_single_score
result_file = result_utils.result_file


MERGE_PAIRS_JSON = result_file("nlp/intermediate_hf/merge_pairs.json")
MERGE_PAIRS_FROM_MNLI_JSON = result_file(
    "nlp/intermediate_hf/merge_pairs_from_mnli.json"
)

# These are 4096 SQuAD examples for the Fisher.
MERGE_FROM_MNLI_PAIRS_SQUAD_DONOR_JSON = result_file(
    "nlp/intermediate_hf/merge_from_mnli_pairs_squad_donor.json"
)
MERGE_SQUAD_DONOR_JSON = result_file("nlp/intermediate_hf/merge_squad_donor.json")
MERGE_SQUAD_DONOR_HIGH_RESOURCE_JSON = result_file(
    "nlp/intermediate_hf/merge_squad_donor_high_resource.json"
)

MERGE_FROM_MNLI_PAIRS_SQUAD_DONOR_1024_JSON = result_file(
    "nlp/intermediate_hf/merge_from_mnli_pairs_squad_donor_1024.json"
)

MERGE_FROM_MNLI_RTE_LARGE_JSON = result_file(
    "nlp/intermediate_hf/merge_from_mnli_rte_large.json"
)


def create_json(merge_exp):
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

        items.append(
            {
                "target_task": target_mtm.task,
                "donor_task": donor_mtm.task,
                "trial_index": params.trial_index,
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
        group_key = hashabledict(
            {
                "target_task": item["target_task"],
                "donor_task": item["donor_task"],
            }
        )
        row_groups[group_key].append(item)

    header = [
        "task",
        "donor",
        "merged score",
        "stddev",
        "orig score",
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
            hp["target_task"],
            hp["donor_task"],
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


def latex_render_score_subscript(mean, stddev, round_digits=1):
    mean = round(mean, round_digits)
    if stddev is None:
        return f"${mean}$"
    stddev = round(stddev, round_digits)
    return f"${mean}_{{{stddev}}}$"


# result_utils.TASK_NICE_NAMES
def create_latex_table(
    filepath,
    render_score_fn=latex_render_score_subscript,
    task_order=result_utils.GLUE_TASKS_ORDER,
):
    items = result_utils.load_json(filepath)

    row_groups = collections.defaultdict(list)
    for item in items:
        group_key = hashabledict(
            {
                "target_task": item["target_task"],
                "donor_task": item["donor_task"],
            }
        )
        row_groups[group_key].append(item)

    def create_donor_to_merge_summary(target_task):
        ret = collections.defaultdict(list)
        for k, v in row_groups.items():
            if k["target_task"] != target_task:
                continue
            ret[k["donor_task"]].extend(v)

        ret2 = {}
        for donor_task, ret_items in ret.items():
            merged_scores = np.array(
                [get_single_score(item["merged_score"]) for item in ret_items]
            )
            mean = np.mean(merged_scores)
            stddev = np.std(merged_scores) if len(ret_items) > 1 else None
            ret2[donor_task] = (mean, stddev)
        return ret2

    rows = [len(task_order) * [""] for _ in task_order]

    for col_idx, target_task in enumerate(task_order):
        donor_to_merge_summary = create_donor_to_merge_summary(target_task)
        for row_idx, donor_task in enumerate(task_order):
            if donor_task not in donor_to_merge_summary:
                continue
            mean, stddev = donor_to_merge_summary[donor_task]
            rows[row_idx][col_idx] = render_score_fn(mean, stddev)

    for row, task in zip(rows, task_order):
        row.insert(0, result_utils.TASK_NICE_NAMES[task])

    rows = [
        R"\toprule",
        [R"\textbf{Task}"] + [result_utils.TASK_NICE_NAMES[t] for t in task_order],
        R"\midrule",
        *rows,
        R"\bottomrule",
    ]

    return result_utils.table_to_latex(rows)


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.intermediate_hf import merge
    from m251.exp_groups.paper.nlp.intermediate_hf import merge_large

    ###########################################################################

    filepath = MERGE_PAIRS_JSON
    t = create_latex_table(filepath)
    print(t)

    ###########################################################################

    # merge_exp = merge_large.Merge_Pairs_Normalized_LastCkpt
    # summary = create_json(merge_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)

    # filepath = MERGE_FROM_MNLI_RTE_LARGE_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_BertBase_HighResource_SquadDonor_4096
    # summary = create_json(merge_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)

    # filepath = MERGE_SQUAD_DONOR_HIGH_RESOURCE_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_BertBaseFromMnli_SquadDonor_1024
    # summary = create_json(merge_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)

    # filepath = MERGE_FROM_MNLI_PAIRS_SQUAD_DONOR_1024_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_BertBase_SquadDonor_4096
    # summary = create_json(merge_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)

    # filepath = MERGE_SQUAD_DONOR_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_BertBaseFromMnli_SquadDonor_4096
    # summary = create_json(merge_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)

    # filepath = MERGE_FROM_MNLI_PAIRS_SQUAD_DONOR_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_BertBaseFromMnli_Pairs
    # summary = create_json(merge_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)

    # filepath = MERGE_PAIRS_FROM_MNLI_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_BertBase_Pairs
    # summary = create_json(merge_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)

    # filepath = MERGE_PAIRS_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath)
    # print(t)
