"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/ablations/fisher_comp2/results/merge_results.py

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


TEMP_JSON = "/tmp/asdfjadslfjsdal.json"


def create_json(merge_exp, target_fisher_exp, donor_fisher_exp):
    with merge_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[
                merge_exp.uuid,
                target_fisher_exp.uuid,
                donor_fisher_exp.uuid,
            ]
        )

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=merge_exp.uuid)

    def get_fisher_examples(mtm, fisher_exp):
        # print(mtm.fisher_run_uuid)
        fisher_run = exps_data.get_run_data(mtm.fisher_run_uuid)
        # print(fisher_run._items)
        fisher_params = fisher_run.get_single_item_by_class(fisher_exp.params_cls)
        return fisher_params.num_examples or NUM_GLUE_TRAIN_EXAMPLES[mtm.task]

    items = []
    for run_id in merge_run_ids:
        merge_run = exps_data.get_run_data(run_id)
        # print(merge_run._items)

        params = merge_run.get_single_item_by_class(merge_exp.params_cls)
        reses = merge_run.get_items_by_class(merging_execs.MergingEvaluationResults)

        res = max(reses, key=lambda r: get_single_score(r.results))
        og_res = max(reses, key=lambda r: r.weighting[0])
        donor_body_res = max(reses, key=lambda r: r.weighting[1])

        assert og_res.weighting[0] == 1.0
        assert donor_body_res.weighting[1] == 1.0

        target_mtm, donor_mtm = params.models_to_merge
        try:
            items.append(
                {
                    "task": target_mtm.task,
                    "other_task": donor_mtm.task,
                    "trial_index": params.trial_index,
                    "hyperparams": {
                        "target_fisher_examples": get_fisher_examples(
                            target_mtm, target_fisher_exp
                        ),
                        "donor_fisher_examples": get_fisher_examples(
                            donor_mtm, donor_fisher_exp
                        ),
                    },
                    "original_score": og_res.results,
                    "merged_score": res.results,
                    "donor_body_score": donor_body_res.results,
                    "weighting": res.weighting[0],
                }
            )
        except AssertionError:
            print("Skipping result because I accidentially did it backwards.")
            continue

    return items


def create_csv_table(filepath, round_digits=1):
    items = result_utils.load_json(filepath)

    row_groups = collections.defaultdict(list)
    for item in items:
        group_key = hashabledict(item["hyperparams"])
        row_groups[group_key].append(item)

    header = [
        "MNLI examples",
        "RTE examples",
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
            hp["donor_fisher_examples"],
            hp["target_fisher_examples"],
            round(np.mean(merged_scores), round_digits),
            round(np.std(merged_scores), round_digits),
            #
            round(np.mean(og_scores), round_digits),
            round(np.std(og_scores), round_digits),
            #
            round(np.mean(donor_body_scores), round_digits),
            round(np.std(donor_body_scores), round_digits),
            len(merged_scores),
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
):
    items = result_utils.load_json(filepath)

    groups = collections.defaultdict(list)
    for item in items:
        hp = item["hyperparams"]
        groups[(hp["donor_fisher_examples"], hp["target_fisher_examples"])].append(item)

    merged_scores = {}
    for k, group_items in groups.items():
        scores = np.array(
            [get_single_score(item["merged_score"]) for item in group_items]
        )
        mean = np.mean(scores)
        stddev = np.std(scores) if len(group_items) > 1 else None
        merged_scores[k] = render_score_fn(mean, stddev)

    all_donor_fisher_examples = sorted(set(k[0] for k in groups.keys()))
    all_target_fisher_examples = sorted(set(k[1] for k in groups.keys()))

    rows = [len(all_target_fisher_examples) * [""] for _ in all_donor_fisher_examples]

    for col_idx, target_examples in enumerate(all_target_fisher_examples):
        for row_idx, donor_examples in enumerate(all_donor_fisher_examples):
            rows[row_idx][col_idx] = merged_scores[(donor_examples, target_examples)]

    for row, examples in zip(rows, all_donor_fisher_examples):
        row.insert(0, str(examples))

    rows = [
        R"\toprule",
        [R"\textbf{Examples}"]
        + [str(examples) for examples in all_target_fisher_examples],
        R"\midrule",
        *rows,
        R"\bottomrule",
    ]

    return result_utils.table_to_latex(rows)


if __name__ == "__main__":
    from m251.exp_groups.paper.ablations.fisher_comp2 import fisher
    from m251.exp_groups.paper.ablations.fisher_comp2 import merge
    from m251.exp_groups.paper.ablations.fisher_comp2 import dummy_merge

    ###########################################################################

    merge_exp = merge.Merge_Rte_Mnli
    # merge_exp = dummy_merge.Merge_Rte_Mnli_Dummy

    target_fisher_exp = fisher.Fisher_BertBase_Rte
    donor_fisher_exp = fisher.Fisher_BertBase_Mnli
    summary = create_json(merge_exp, target_fisher_exp, donor_fisher_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)
    filepath = TEMP_JSON
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    # t = create_csv_table(filepath)
    t = create_latex_table(filepath)
    print(t)
