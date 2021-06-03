"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/ablations/ckpt_choice/results/merge_results.py

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


MERGE_JSON = result_file("ablations/ckpt_choice/merge.json")
MERGE_RTE_10_EPOCHS_JSON = result_file("ablations/ckpt_choice/merge_rte_10_epochs.json")


def create_json(merge_exp):
    with merge_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[merge_exp.uuid])

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=merge_exp.uuid)

    items = []
    for run_id in merge_run_ids:
        merge_run = exps_data.get_run_data(run_id)

        params = merge_run.get_single_item_by_class(merge_exp.params_cls)
        reses = merge_run.get_items_by_class(merging_execs.MergingEvaluationResults)

        res = max(reses, key=lambda r: get_single_score(r.results))
        og_res = max(reses, key=lambda r: r.weighting[0])
        donor_body_res = max(reses, key=lambda r: r.weighting[1])

        assert og_res.weighting[0] == 1.0
        assert donor_body_res.weighting[1] == 1.0

        target_mtm, donor_mtm = params.models_to_merge
        items.append(
            {
                "task": target_mtm.task,
                "other_task": donor_mtm.task,
                "trial_index": params.trial_index,
                "hyperparams": {
                    "target_ckpt_index": params.target_ckpt_index,
                    "donor_ckpt_index": params.donor_ckpt_index,
                },
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

    header = [
        "MNLI ckpt",
        "RTE ckpt",
        "merged RTE acc",
        "merged stddev",
        "orig RTE acc",
        "orig stddev",
        "MNLI body acc",
        "MNLI body stddev",
        "num trials",
    ]
    body = []
    for hp, row_items in row_groups.items():
        og_scores = [get_single_score(item["original_score"]) for item in row_items]
        merged_scores = [get_single_score(item["merged_score"]) for item in row_items]
        donor_body_scores = [
            get_single_score(item["donor_body_score"]) for item in row_items
        ]
        row = [
            hp["donor_ckpt_index"],
            hp["target_ckpt_index"],
            round(np.mean(merged_scores), round_digits),
            round(np.std(merged_scores), round_digits),
            #
            round(np.mean(og_scores), round_digits),
            round(np.std(og_scores), round_digits),
            #
            round(np.mean(donor_body_scores), round_digits),
            round(np.std(donor_body_scores), round_digits),
            len(row_items),
        ]
        body.append(row)

    body = sorted(body, key=lambda r: r[:2])

    rows = [header] + body

    return result_utils.csv_to_str(rows)


def latex_render_score_subscript(mean, stddev, round_digits=1, is_orig=False):
    if mean is None:
        return "---"

    mean = round(mean, round_digits)
    if stddev is None:
        ret = f"{mean}"
    else:
        stddev = round(stddev, round_digits)
        ret = f"{mean}_{{{stddev}}}"

    if is_orig:
        ret = f"\\mathit{{{ret}}}"

    return f"${ret}$"


def _index_to_epoch(item, filepath):
    hp = item["hyperparams"]
    donor_epoch = (hp["donor_ckpt_index"] + 1) / 2
    if "10" in filepath:
        target_epoch = float(hp["target_ckpt_index"] + 1)
    else:
        target_epoch = (hp["target_ckpt_index"] + 1) / 2
    item = item.copy()
    item["hyperparams"] = {
        "target_epoch": target_epoch,
        "donor_epoch": donor_epoch,
    }
    return item


def create_latex_table(  # noqa: C901
    filepath,
    render_score_fn=latex_render_score_subscript,
):
    if not isinstance(filepath, (list, tuple)):
        filepath = [filepath]

    items = []
    for fp in filepath:
        its = result_utils.load_json(fp)
        its = [_index_to_epoch(it, fp) for it in its]
        its = [it for it in its if it["hyperparams"]["target_epoch"].is_integer()]
        items.extend(its)

    groups = collections.defaultdict(list)
    for item in items:
        hp = item["hyperparams"]
        groups[(hp["donor_epoch"], hp["target_epoch"])].append(item)

    original_scores = {}
    for target_epoch in range(1, 11):
        group_items = max(
            [
                group_items
                for (_, te), group_items in groups.items()
                if te == target_epoch
            ],
            key=len,
        )
        og_scores = np.array(
            [get_single_score(item["original_score"]) for item in group_items]
        )
        mean = np.mean(og_scores)
        stddev = np.std(og_scores) if len(group_items) > 1 else None
        original_scores[target_epoch] = render_score_fn(mean, stddev)

    merged_scores = {}
    for k, group_items in groups.items():
        scores = np.array(
            [get_single_score(item["merged_score"]) for item in group_items]
        )
        og_mean = np.mean(
            [get_single_score(item["original_score"]) for item in group_items]
        )
        mean = np.mean(scores)
        stddev = np.std(scores) if len(group_items) > 1 else None
        merged_scores[k] = render_score_fn(mean - og_mean, stddev)

    rows = []
    for row_idx in range(8):
        donor_epoch = (row_idx + 1) / 2
        rows.append(
            [str(donor_epoch)]
            + [
                merged_scores[(donor_epoch, float(target_epoch))]
                for target_epoch in range(1, 11)
            ]
        )

    rows = [
        R"\toprule",
        [R"\textbf{Epoch}"] + [str(target_epoch) for target_epoch in range(1, 11)],
        R"\midrule",
        [R"\textit{Unmerged}"]
        + [original_scores[target_epoch] for target_epoch in range(1, 11)],
        R"\midrule",
        *rows,
        R"\bottomrule",
    ]

    return result_utils.table_to_latex(rows)


if __name__ == "__main__":
    from m251.exp_groups.paper.ablations.ckpt_choice import fisher
    from m251.exp_groups.paper.ablations.ckpt_choice import merge

    ###########################################################################
    # filepath = [MERGE_JSON, MERGE_RTE_10_EPOCHS_JSON]
    # t = create_latex_table(filepath)
    # print(t)

    ###########################################################################
    ###########################################################################
    merge_exp = merge.Merge
    # merge_exp = merge.DummyMerge
    summary = create_json(merge_exp)

    filepath = "/tmp/kfgsdkfdg.json"
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    t = create_csv_table(filepath)
    print(t)

    ###########################################################################

    # merge_exp = merge.Merge_Rte10Epochs
    # summary = create_json(merge_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)
    # with open(MERGE_RTE_10_EPOCHS_JSON, "w") as f:
    #     json.dump(summary, f, indent=2)

    ###########################################################################

    # filepath = MERGE_JSON
    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge
    # summary = create_json(merge_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)
    # with open(MERGE_JSON, "w") as f:
    #     json.dump(summary, f, indent=2)
