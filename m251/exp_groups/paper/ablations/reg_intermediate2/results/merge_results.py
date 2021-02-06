"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/ablations/reg_intermediate2/results/merge_results.py

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


def create_json(merge_exp, fisher_exps):
    with merge_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[merge_exp.uuid] + [e.uuid for e in fisher_exps]
        )

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=merge_exp.uuid)

    def get_reg_str(mtm):
        fisher_run = exps_data.get_run_data(mtm.fisher_run_uuid)
        for exp in fisher_exps:
            try:
                fisher_params = fisher_run.get_single_item_by_class(exp.params_cls)
                return fisher_params.reg_strength
            except AssertionError:
                continue

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
                    "target_reg_strength": get_reg_str(target_mtm),
                    "donor_reg_strength": get_reg_str(donor_mtm),
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
        "MNLI λ",
        "RTE λ",
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
            hp["donor_reg_strength"],
            hp["target_reg_strength"],
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


def create_latex_table(  # noqa: C901
    filepath,
    render_score_fn=latex_render_score_subscript,
    l2_coeffs=(0.0, 1e-6, 3e-4, 0.01, 0.1),
    coeff_to_pretty={
        0.0: "0",
        1e-6: "1e-6",
        3e-4: "3e-4",
        0.01: "1e-2",
        0.1: "1e-1",
    },
):
    items = result_utils.load_json(filepath)

    row_groups = collections.defaultdict(list)
    for item in items:
        group_key = hashabledict(item["hyperparams"])
        row_groups[group_key].append(item)

    def get_original_score(target_coeff):
        ret = {}
        for k, v in row_groups.items():
            if k["target_reg_strength"] != target_coeff:
                continue
            ret[k["donor_reg_strength"]] = v

        if not ret:
            return None, None

        ret_items = max(ret.values(), key=len)
        merged_scores = np.array(
            [get_single_score(item["original_score"]) for item in ret_items]
        )
        mean = np.mean(merged_scores)
        stddev = np.std(merged_scores) if len(ret_items) > 1 else None

        return render_score_fn(mean, stddev)

    rows = [len(l2_coeffs) * [""] for _ in l2_coeffs]

    for col_idx, target_coeff in enumerate(l2_coeffs):
        for row_idx, donor_coeff in enumerate(l2_coeffs):
            key = hashabledict(
                {
                    "target_reg_strength": target_coeff,
                    "donor_reg_strength": donor_coeff,
                }
            )
            row_items = row_groups[key]
            merged_scores = np.array(
                [get_single_score(item["merged_score"]) for item in row_items]
            )
            mean = np.mean(merged_scores)
            stddev = np.std(merged_scores) if len(row_items) > 1 else None
            rows[row_idx][col_idx] = render_score_fn(mean, stddev)

    for row, coeff in zip(rows, l2_coeffs):
        row.insert(0, coeff_to_pretty[coeff])

    rows = [
        R"\toprule",
        [""] + [coeff_to_pretty[t] for t in coeff_to_pretty],
        R"\midrule",
        ["Original"] + [get_original_score(t) for t in coeff_to_pretty],
        R"\midrule",
        *rows,
        R"\bottomrule",
    ]

    return result_utils.table_to_latex(rows)


if __name__ == "__main__":
    from m251.exp_groups.paper.ablations.reg_intermediate2 import fisher
    from m251.exp_groups.paper.ablations.reg_intermediate2 import merge

    ###########################################################################

    merge_exp = merge.Merge
    fisher_exps = [fisher.Fisher_Rte, fisher.Fisher_Mnli]
    summary = create_json(merge_exp, fisher_exps)
    # s = json.dumps(summary, indent=2)
    # print(s)
    with open(TEMP_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    filepath = TEMP_JSON
    t = create_latex_table(filepath)
    print(t)
