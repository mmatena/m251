"""TODO: Reorganize and rename.


OLD JSON schema:
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


NEW JSON schema:
[
    {
        'task': str,
        'other_task': str,
        'hyperparams': {
            'pretrained_model': str,
            'reg_strength': float,
            'reg_type': str,
        },
        # NOTE: MNLI will have {[split]: {[metric_name]: value}} instead
        # for scores.
        'original_score': {[metric_name]: value},
        'merged_score': {[metric_name]: value},
        'weighting': float,
    }
]


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/analyze_merging_json.py


python3 -i m251/exp_groups/bert_merging_prelims/results/analyze_merging_json.py

"""
import collections
import json
import os


GLUE_TASKS = ("cola", "mnli", "mrpc", "sst2", "stsb", "qqp", "qnli", "rte")

# NOTE: These are median of 5 runs.
# NOTE: I'm not sure how much hyperparameter tuning went into these.
# NOTE: I think RTE, STS and MRPC were finetuned starting from the MNLI model.
PAPER_ROBERTA_LARGE_DEV_SCORES = {
    "mnli": 90.2,
    "qnli": 94.7,
    "qqp": 92.2,
    "rte": 86.6,
    "sst2": 96.4,
    "mrpc": 90.9,
    "cola": 68.0,
    "stsb": 92.4,
}

BERT_PRELIM_BASE_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_prelim_base.json"
)

BERT_PRELIM_LARGE_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_prelim_large.json"
)

ROBERTA_LARGE_PRELIM_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/roberta_large_prelim.json"
)


def _load_json(json_file):
    json_file = os.path.expanduser(json_file)
    with open(json_file, "r") as f:
        return json.load(f)


# def get_best_absolute_scores_for_task(json_file, task, k=1):
#     assert k == 1, "TODO: Handle k != 1."
#     items = _load_json(json_file)

#     best_score = 0
#     best_item = None

#     for item in items:
#         if (
#             task in item["original_scores"]
#             and item["original_scores"][task] > best_score
#         ):
#             best_score = item["original_scores"][task]
#             best_item = {
#                 "common_parameters": item["common_parameters"],
#                 "score": best_score,
#                 "is_merged": False,
#             }
#         for result in item["merge_results"]:
#             if task in result["scores"] and result["scores"][task] > best_score:
#                 best_score = result["scores"][task]
#                 other_task, = [k for k in result["scores"].keys() if k != task]
#                 best_item = {
#                     "common_parameters": item["common_parameters"],
#                     "score": best_score,
#                     "is_merged": True,
#                     "original_score": item["original_scores"][task],
#                     "absolute_difference": result["absolute_difference"][task],
#                     "other_task": other_task,
#                     'weighting': result['weighting'][task],
#                 }
#     return best_item


# def get_best_unmerged_scores_for_task(json_file, task, k=1):
#     assert k == 1, "TODO: Handle k != 1."
#     items = _load_json(json_file)

#     best_score = 0
#     best_item = None

#     for item in items:
#         if (
#             task in item["original_scores"]
#             and item["original_scores"][task] > best_score
#         ):
#             best_score = item["original_scores"][task]
#             best_item = {
#                 "common_parameters": item["common_parameters"],
#                 "score": best_score,
#                 "is_merged": False,
#             }
#     return best_item


def _get_single_score(scores):
    if not isinstance(scores, dict):
        return scores
    values = [_get_single_score(v) for v in scores.values()]
    return sum(values) / len(values)


def get_best_absolute_scores_for_task(json_file, task, k=1):
    assert k == 1, "TODO: Handle k != 1."
    items = _load_json(json_file)

    best_score = 0
    best_item = None
    best_key = None

    for item in items:
        if item["task"] != task:
            continue
        for key in ["original_score", "merged_score"]:
            score = _get_single_score(item[key])
            if score > best_score:
                best_score = score
                best_item = item
                best_key = key

    return {
        # Essential info.
        "task": task,
        "score": best_score,
        "is_merged": best_key == "merged_score",
        # Helpful info.
        "item": best_item,
        "key": best_key,
    }


def get_best_unmerged_scores_for_task(json_file, task, k=1):
    assert k == 1, "TODO: Handle k != 1."
    items = _load_json(json_file)

    best_score = 0
    best_item = None

    for item in items:
        if item["task"] != task:
            continue
        score = _get_single_score(item["original_score"])
        if score > best_score:
            best_score = score
            best_item = item

    return {
        # Essential info.
        "task": task,
        "score": best_score,
        "is_merged": False,
        # Helpful info.
        "item": best_item,
        "key": "original_score",
    }


def get_merging_single_task_summary(json_file, tasks=GLUE_TASKS):
    ret = {}
    for task in tasks:
        best_overall = get_best_absolute_scores_for_task(json_file, task)
        best_unmerged = get_best_unmerged_scores_for_task(json_file, task)

        best_item = best_overall["item"]

        summary = {
            "best_score": best_overall["score"],
            "roberta_paper_score": PAPER_ROBERTA_LARGE_DEV_SCORES[task],
            # 'best_score_full': best_item[best_overall['key']],
            "is_merged": best_overall["is_merged"],
            # 'hyperparams': best_item['hyperparams']
            "reg_strength": best_item["hyperparams"]["reg_strength"],
        }
        if best_overall["score"] > best_unmerged["score"]:
            # Best is merged.
            summary.update(
                {
                    "other_task": best_item["other_task"],
                    "weighting": best_item["weighting"],
                    "absolute_difference": best_overall["score"]
                    - best_unmerged["score"],
                }
            )

        ret[task] = summary
    return ret


###############################################################################
###############################################################################

"""
           MNLI      QNLI QQP  RTE  SST  MRPC CoLA STS  WNLI
Single-task single models on dev
BERTLARGE  86.6/-    92.3 91.3 70.4 93.2 88.0 60.6 90.0 -
XLNetLARGE 89.8/-    93.9 91.8 83.8 95.6 89.2 63.6 91.8 -
RoBERTa    90.2/90.2 94.7 92.2 86.6 96.4 90.9 68.0 92.4 91.3
"""


if True:
    json_file_path = ROBERTA_LARGE_PRELIM_JSON
    summary = get_merging_single_task_summary(json_file_path)
    average_improvement = sum(
        v.get("absolute_difference", 0.0) for v in summary.values()
    ) / len(summary.values())

    summary = {
        "summary": summary,
        "average_improvement": average_improvement,
    }

    s = json.dumps(summary, indent=2)
    print(s)

    # out = {}
    # # json_file_path = BERT_PRELIM_BASE_JSON
    # json_file_path = BERT_PRELIM_LARGE_JSON
    # for task in ["qqp", "sst2", "qnli", "mrpc", "rte"]:
    #     best_overall = get_best_absolute_scores_for_task(json_file_path, task)
    #     best_unmerged = get_best_unmerged_scores_for_task(json_file_path, task)

    #     # best_overall['best_unmerged'] = best_unmerged
    #     # if best_overall['is_merged']:
    #     #     best_overall['absolute_difference'] = best_overall['score'] - best_unmerged['score']

    #     ret = {
    #         "best_score": 100 * best_overall["score"],
    #         "best_unmerged_score": 100 * best_unmerged["score"],
    #         "absolute_difference": 100
    #         * (best_overall["score"] - best_unmerged["score"]),
    #         # 'best_is_merged': best_overall['is_merged'],
    #     }
    #     if best_overall['is_merged']:
    #         ret['other_task'] = best_overall['other_task']
    #         ret['weighting'] = best_overall['weighting']
    #     out[task] = ret
    # s = json.dumps(out, indent=2)
    # print(s)
