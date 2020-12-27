"""TODO: Reorganize and rename."

JSON schema:
[
    {
        'common_parameters': {
            'pretrained_model': str,
            'reg_strength': float,
            'reg_type': str,
        }
        'original_scores': {[task]: accuracy}
        'merge_results': [
            {
                'weighting': {[task]: task_weight},
                'scores': {[task]: accuracy},
                'relative_difference': {[task]: rel_diff},
                'absolute_difference': {[task]: abs_diff},
            }
        ],
    }
]


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 -i m251/exp_groups/bert_merging_prelims/results/analyze_merging_json.py

"""
import collections
import json
import os


BERT_PRELIM_BASE_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_prelim_base.json"
)


def _load_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def get_best_absolute_scores_for_task(json_file, task, k=1):
    assert k == 1, "TODO: Handle k != 1."
    items = _load_json(json_file)

    best_score = 0
    best_item = None

    for item in items:
        if (
            task in item["original_scores"]
            and item["original_scores"][task] > best_score
        ):
            best_score = item["original_scores"][task]
            best_item = {
                "common_parameters": item["common_parameters"],
                "score": best_score,
                "is_merged": False,
            }
        for result in item["merge_results"]:
            if task in result["scores"] and result["scores"][task] > best_score:
                best_score = result["scores"][task]
                best_item = {
                    "common_parameters": item["common_parameters"],
                    "score": best_score,
                    "is_merged": True,
                    "original_score": item["original_scores"][task],
                    "absolute_difference": result["absolute_difference"][task],
                }
    return best_item


def get_best_unmerged_scores_for_task(json_file, task, k=1):
    assert k == 1, "TODO: Handle k != 1."
    items = _load_json(json_file)

    best_score = 0
    best_item = None

    for item in items:
        if (
            task in item["original_scores"]
            and item["original_scores"][task] > best_score
        ):
            best_score = item["original_scores"][task]
            best_item = {
                "common_parameters": item["common_parameters"],
                "score": best_score,
                "is_merged": False,
            }
    return best_item


###############################################################################
###############################################################################


_OG_PAPER_BERT_BASE_TEST_ACCURACIES = {
    "qnli": 90.5,
    "ret": 66.4,
    "sst2": 93.5,
}

if True:
    out = {}
    for task in ["qqp", "sst2", "qnli", "mrpc", "rte"]:
        best_overall = get_best_absolute_scores_for_task(BERT_PRELIM_BASE_JSON, task)
        best_unmerged = get_best_unmerged_scores_for_task(BERT_PRELIM_BASE_JSON, task)

        # best_overall['best_unmerged'] = best_unmerged
        # if best_overall['is_merged']:
        #     best_overall['absolute_difference'] = best_overall['score'] - best_unmerged['score']

        ret = {
            "best_score": 100 * best_overall["score"],
            "best_unmerged_score": 100 * best_unmerged["score"],
            "absolute_difference": 100
            * (best_overall["score"] - best_unmerged["score"]),
            # 'best_is_merged': best_overall['is_merged'],
        }

        out[task] = ret
    s = json.dumps(out, indent=2)
    print(s)
