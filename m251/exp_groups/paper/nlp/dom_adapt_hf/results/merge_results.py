"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/dom_adapt_hf/results/merge_results.py

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


MERGE_MLM_S2ORC_131072_JSON = result_file(
    "nlp/dom_adapt_hf/merge_mlm_s2orc_131072.json"
)

MERGE_DAPT_MLM_S2ORC_131072_JSON = result_file(
    "nlp/dom_adapt_hf/merge_dapt_mlm_s2orc_131072.json"
)
MERGE_DAPT_MLM_S2ORC_4096_JSON = result_file(
    "nlp/dom_adapt_hf/merge_dapt_mlm_s2orc_4096.json"
)


MERGE_MLM_S2ORC_131072_TEST_JSON = result_file(
    "nlp/dom_adapt_hf/merge_mlm_s2orc_131072_test.json"
)

MERGE_DAPT_MLM_S2ORC_131072_TEST_JSON = result_file(
    "nlp/dom_adapt_hf/merge_dapt_mlm_s2orc_131072_test.json"
)


MERGE2_ROBERTA_MLM_S2ORC_131072_VALIDATION_JSON = result_file(
    "nlp/dom_adapt_hf/merge2_roberta_mlm_s2orc_131072_validation.json"
)

MERGE2_ROBERTA_MLM_S2ORC_131072_TEST_JSON = result_file(
    "nlp/dom_adapt_hf/merge2_roberta_mlm_s2orc_131072_test.json"
)

MERGE2_ROBERTA_MLM_S2ORC_131072_TEST_CKPT_5_JSON = result_file(
    "nlp/dom_adapt_hf/merge2_roberta_mlm_s2orc_131072_test_ckpt_5.json"
)

MERGE2_ROBERTA_MLM_S2ORC_131072_TEST_CKPT_9_JSON = result_file(
    "nlp/dom_adapt_hf/merge2_roberta_mlm_s2orc_131072_test_ckpt_9.json"
)

MERGE2_ROBERTA_MLM_S2ORC_131072_TEST_CKPT_9_HALF_WEIGHT_JSON = result_file(
    "nlp/dom_adapt_hf/merge2_roberta_mlm_s2orc_131072_test_ckpt_9_half_weight.json"
)

MERGE2_ROBERTA_ORIGINAL_TEST_JSON = result_file(
    "nlp/dom_adapt_hf/merge2_roberta_original_test.json"
)


TEMP_JSON = "/tmp/merge_temp.json"


def create_json(
    merge_exp,
    group_by_ckpt_index=True,
    group_by_finetuned_model=False,
    group_by_donor_fisher=False,
):
    with merge_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[merge_exp.uuid])

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=merge_exp.uuid)
    print(len(merge_run_ids))
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
        }
        if group_by_ckpt_index:
            hyperparams["target_ckpt_index"] = params.target_ckpt_index
        if group_by_finetuned_model:
            hyperparams["train_run_uuid"] = target_mtm.train_run_uuid
        if group_by_donor_fisher:
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
            # best = max(row_items, key=lambda r: get_single_score(r["original_score"]))

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

        # q = (FISHER_RUN_UUID_TO_INFO[hp['donor_fisher']])
        # q = f'{q[0]}, {q[1]}'
        # print(np.max(merged_scores))
        # print(np.min(merged_scores))
        # print(np.max(og_scores))
        # print(np.min(og_scores))
        # print('\n')
        row = [
            hp["task"],
            hp["target_ckpt_index"] if group_by_ckpt_index else "-",
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

    body = sorted(body, key=lambda r: r[:2])

    rows = [header] + body

    return result_utils.csv_to_str(rows)


PRETRAIN_RUN_UUID_TO_NUM_EXAMPLES = {
    "71a3f6b94b724e358cc90855d82d6827": 32768,
    "15ca7d2f8ac14190831083092cdce5f2": 262144,
    "2ffef6eb797e4994ac283fe1fd75dd65": 2097152,
}

# (pretrain_examples, fisher_examples)
FISHER_RUN_UUID_TO_INFO = {
    "ea71326947744b949dc63e41eef092a9": (32768, 16384),
    "0710ed2634e14792ad5ce5978191fcd9": (32768, 131072),
    "573c7d86dbc349d0b715257fb340ce45": (262144, 16384),
    "3acd2931b19d4d72b63afd08cef1f605": (262144, 131072),
    "d6adb8a7082a4edfb2449d3ed535f6dc": (2097152, 16384),
    "7bc2a5fec97647238b7b9fa5a2bb47ca": (2097152, 131072),
}


# FISHER_UUID_TO_INFO = {
#     #
# }


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.dom_adapt_hf import eval_test_set
    from m251.exp_groups.paper.nlp.dom_adapt_hf import eval_test_set2
    from m251.exp_groups.paper.nlp.dom_adapt_hf import merge
    from m251.exp_groups.paper.nlp.dom_adapt_hf import merge2
    from m251.exp_groups.paper.nlp.dom_adapt_hf import merge4
    from m251.exp_groups.paper.nlp.dom_adapt_hf import merge5

    ###########################################################################

    # merge_exp = merge5.Merge_ROBERTA_LastCkpt_TestSet_PretrainedMore_REAL
    # merge_exp = merge5.Merge_ROBERTA_LastCkpt_TestSet_PretrainFromDapt32768
    merge_exp = merge5.Merge_ROBERTA_LastCkpt_TestSet_Pretrain32768NoReg

    summary = create_json(
        merge_exp,
        group_by_ckpt_index=False,
        group_by_finetuned_model=True,
        group_by_donor_fisher=True,
    )
    filepath = TEMP_JSON
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)

    t = create_csv_table(
        filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    )
    print(t)

    ###########################################################################

    # # merge_exp = merge2.Merge_ROBERTA_LastCkpt_TestSet
    # # merge_exp = merge2.Merge_ROBERTA_LastCkpt_WrongMerge_TestSet
    # merge_exp = merge4.Merge_ROBERTA_LastCkpt_TestSet_WithDaptAllVars
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = TEMP_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = merge4.Merge_ROBERTA_LastCkpt_TestSet_WithDaptAllVars_MergeOnlyBody
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = TEMP_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = eval_test_set2.EvalTest_Merged_MlmS2orc_ROBERTA_Ckpt9_HalfWeight
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = MERGE2_ROBERTA_MLM_S2ORC_131072_TEST_CKPT_9_HALF_WEIGHT_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = eval_test_set2.EvalTest_Merged_MlmS2orc_ROBERTA_Ckpt9
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = MERGE2_ROBERTA_MLM_S2ORC_131072_TEST_CKPT_9_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = eval_test_set2.EvalTest_Merged_MlmS2orc_ROBERTA_Ckpt5
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = MERGE2_ROBERTA_MLM_S2ORC_131072_TEST_CKPT_5_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = eval_test_set2.EvalTest_Original_MlmS2orc_ROBERTA_Best
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = MERGE2_ROBERTA_ORIGINAL_TEST_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = eval_test_set2.EvalTest_Merged_MlmS2orc_ROBERTA_Best
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = MERGE2_ROBERTA_MLM_S2ORC_131072_TEST_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = merge2.Merge_MlmS2orc_ROBERTA
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = MERGE2_ROBERTA_MLM_S2ORC_131072_VALIDATION_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = eval_test_set.Merge_Dapt_MlmS2orc_Normalized_131072_FOR_REAL_Best_Merged
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = MERGE_DAPT_MLM_S2ORC_131072_TEST_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = eval_test_set.Merge_MlmS2orc_Normalized_131072_FOR_REAL_Best_Merged
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = MERGE_MLM_S2ORC_131072_TEST_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_MlmS2orc_Normalized_131072_FOR_REAL
    # summary = create_json(
    #     merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True
    # )
    # filepath = MERGE_MLM_S2ORC_131072_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(
    #     filepath, group_by_ckpt_index=False, best_per_finetuned_model=True
    # )
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_MlmS2orc_Normalized_Mlm4096
    # summary = create_json(merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True)
    # filepath = MERGE_DAPT_MLM_S2ORC_4096_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath, group_by_ckpt_index=False, best_per_finetuned_model=True)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_DAPT_MlmS2orc_Normalized
    # summary = create_json(merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True)
    # filepath = MERGE_MLM_S2ORC_131072_JSON
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath, group_by_ckpt_index=False, best_per_finetuned_model=True)
    # print(t)

    ###########################################################################

    # merge_exp = merge.Merge_MlmS2orc_Normalized
    # summary = create_json(merge_exp, group_by_ckpt_index=False, group_by_finetuned_model=True)
    # filepath = "/tmp/best_original_scores.json"
    # with open(filepath, "w") as f:
    #     json.dump(summary, f, indent=2)

    # t = create_csv_table(filepath, group_by_ckpt_index=False, best_per_finetuned_model=True)
    # print(t)
