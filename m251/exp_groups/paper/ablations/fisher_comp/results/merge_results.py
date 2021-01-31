"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/ablations/fisher_comp/results/merge_results.py

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


VARIATIONAL_JSON = result_file("ablations/fisher_comp/variational.json")
DIRECT_JSON = result_file("ablations/fisher_comp/direct.json")
DIRECT_RTE_10_EPOCH_JSON = result_file("ablations/fisher_comp/direct_rte_10_epoch.json")
DUMMY_RTE_10_EPOCH_JSON = result_file("ablations/fisher_comp/dummy_rte_10_epoch.json")


def create_json(merge_exp, target_fisher_exp, donor_fisher_exp=None):
    if not donor_fisher_exp:
        donor_fisher_exp = target_fisher_exp
    with merge_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=tuple(
                {merge_exp.uuid, target_fisher_exp.uuid, donor_fisher_exp.uuid}
            )
        )

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=merge_exp.uuid)

    def get_fisher_examples(mtm, fisher_exp):
        fisher_run = exps_data.get_run_data(mtm.fisher_run_uuid)
        fisher_params = fisher_run.get_single_item_by_class(fisher_exp.params_cls)
        return fisher_params.num_examples or NUM_GLUE_TRAIN_EXAMPLES[mtm.task]

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


if __name__ == "__main__":
    from m251.exp_groups.paper.ablations.fisher_comp import fisher
    from m251.exp_groups.paper.ablations.fisher_comp import merge
    from m251.exp_groups.paper.ablations.fisher_comp import merge2

    ###########################################################################

    # merge_exp = merge2.MergeDirectFishers_Dummy
    # target_fisher_exp = fisher.FisherComputation_Rte_10Epochs
    # donor_fisher_exp = fisher.DirectFisherComputation
    # summary = create_json(merge_exp, target_fisher_exp, donor_fisher_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)
    # filepath = DUMMY_RTE_10_EPOCH_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)
    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    merge_exp = merge2.MergeDirectFishers
    target_fisher_exp = fisher.FisherComputation_Rte_10Epochs
    donor_fisher_exp = fisher.DirectFisherComputation
    summary = create_json(merge_exp, target_fisher_exp, donor_fisher_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)
    filepath = DIRECT_RTE_10_EPOCH_JSON
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    t = create_csv_table(filepath)
    print(t)

    ###########################################################################

    # print("\nVariational:")
    # filepath = VARIATIONAL_JSON
    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # print("\nDirect:")
    # filepath = DIRECT_JSON
    # t = create_csv_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge.MergeVariationalFishers
    # fisher_exp = fisher.VariationalFisherComputation
    # summary = create_json(merge_exp, fisher_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)
    # with open(VARIATIONAL_JSON, "w") as f:
    #     json.dump(summary, f, indent=2)

    ###########################################################################

    # merge_exp = merge.MergeDirectFishers
    # fisher_exp = fisher.DirectFisherComputation
    # summary = create_json(merge_exp, fisher_exp)
    # # s = json.dumps(summary, indent=2)
    # # print(s)
    # with open(DIRECT_JSON, "w") as f:
    #     json.dump(summary, f, indent=2)
