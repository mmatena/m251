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

from del8.core.utils.type_util import hashabledict


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

BERT_BASE_EWC_PHASE_I_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_ewc_phase_i.json"
)

BERT_BASE_REGS_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_regs.json"
)

BERT_BASE_REGS_MERGE_INFORMED_PAIRS_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_regs_merge_informed_pairs.json"
)

###############################################################################


def _load_json(json_file):
    json_file = os.path.expanduser(json_file)
    with open(json_file, "r") as f:
        return json.load(f)


###############################################################################


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


###############################################################################


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


def get_hyperparams_to_original_score(json_file, task):
    items = _load_json(json_file)

    hp_to_score = {}

    for item in items:
        if item["task"] != task:
            continue

        hp = hashabledict(item["hyperparams"])
        hp_to_score[hp] = item["original_score"]

    return hp_to_score


def get_hyperparams_to_best_merged_score(json_file, task):
    items = _load_json(json_file)

    hp_to_score = collections.defaultdict(list)

    for item in items:
        if item["task"] != task:
            continue

        hp = hashabledict(item["hyperparams"])
        hp_to_score[hp].append(item["merged_score"])

    hp_to_score = {
        hp: max(scores, key=lambda s: _get_single_score(s))
        for hp, scores in hp_to_score.items()
    }

    return hp_to_score


def get_regularization_effect_on_original_score_summary(
    json_file, tasks=GLUE_TASKS, get_hp_to_score_fn=get_hyperparams_to_original_score
):
    # Assumes that the pretrained_model is always the same.

    ret = {}

    for task in tasks:
        hp_to_score = get_hp_to_score_fn(json_file, task)

        items = sorted(
            hp_to_score.items(), key=lambda x: (x[0]["reg_type"], x[0]["reg_strength"])
        )

        task_ret = {}
        reg_types = set(hp["reg_type"] for hp in hp_to_score.keys())
        for reg_type in sorted(reg_types):
            reg_type_ret = collections.defaultdict(list)

            subitems = [x for x in items if x[0]["reg_type"] == reg_type]
            for hp, score in subitems:
                # for key in score:
                #     reg_type_ret[key].append(score[key])
                reg_type_ret["reg_strength"].append(hp["reg_strength"])
                reg_type_ret["single_score"].append(_get_single_score(score))

            task_ret[reg_type] = reg_type_ret
        ret[task] = task_ret

    return ret


def get_regularization_effect_on_rel_merge_performance_summary(
    json_file,
    reg_type,
    tasks=GLUE_TASKS,
):
    # Assumes that the pretrained_model is always the same.
    ret = {}

    task_to_og_summary = get_regularization_effect_on_original_score_summary(
        json_file, tasks, get_hp_to_score_fn=get_hyperparams_to_original_score
    )

    task_to_merge_summary = get_regularization_effect_on_original_score_summary(
        json_file, tasks, get_hp_to_score_fn=get_hyperparams_to_best_merged_score
    )

    for task in tasks:
        og_summary = task_to_og_summary[task][reg_type]
        merge_summary = task_to_merge_summary[task][reg_type]

        assert og_summary["reg_strength"] == merge_summary["reg_strength"]

        og_scores = og_summary["single_score"]
        merge_scores = merge_summary["single_score"]

        rel_perfs = []
        for og_score, merge_score in zip(og_scores, merge_scores):
            if not og_score:
                rel_perf = -1.0
            else:
                rel_perf = 100 * merge_score / og_score
            rel_perfs.append(rel_perf)

        ret[task] = {
            reg_type: {
                "reg_strength": og_summary["reg_strength"],
                "single_score": rel_perfs,
            },
        }

    return ret


def reg_effect_summary_to_md_table(summary, reg_type):
    task_results = collections.defaultdict(dict)
    for task, infos in summary.items():
        info = infos[reg_type]
        for score, reg_str in zip(info["single_score"], info["reg_strength"]):
            task_results[reg_str][task] = score

    tasks = sorted(summary.keys())
    reg_strs = sorted(task_results.keys())
    rows = [
        "|" + ("|".join([f"*{reg_type}*"] + tasks)) + "|",
        "|" + ((len(tasks) + 1) * "---|"),
    ]
    for reg_str in reg_strs:
        row = [f"**_{reg_str}_**"]
        for task in tasks:
            score = task_results[reg_str][task]
            score = str(round(score, 1))
            row.append(score)

        row = "|".join(row)
        row = f"|{row}|"
        rows.append(row)

    return "\n".join(rows)


def reg_effect_summary_to_md_table_flipped(summary, reg_type):
    task_results = collections.defaultdict(dict)
    for task, infos in summary.items():
        info = infos[reg_type]
        for score, reg_str in zip(info["single_score"], info["reg_strength"]):
            task_results[reg_str][task] = score

    tasks = sorted(summary.keys())
    reg_strs = sorted(task_results.keys())

    rows = [
        "|" + ("|".join(["*task*"] + [str(r) for r in reg_strs])) + "|",
        "|" + ((len(reg_strs) + 1) * "---|"),
    ]
    for task in tasks:
        row = [f"**_{task}_**"]
        for reg_str in reg_strs:
            score = task_results[reg_str][task]
            score = str(round(score, 1))
            row.append(score)

        row = "|".join(row)
        row = f"|{row}|"
        rows.append(row)

    return "\n".join(rows)


###############################################################################


def get_regularization_effect_on_phase_i_score_summary(
    json_file, results_key, task=None, reg_type=None
):
    # Assumes everything is the same task, model, etc. Only the reg_strength changes.
    items = _load_json(json_file)

    if task is None:
        task = items[0]["hyperparams"]["task"]
    if reg_type is None:
        reg_type = items[0]["hyperparams"]["reg_type"]

    items = sorted(items, key=lambda item: item["hyperparams"]["reg_strength"])

    reg_strengths = []
    single_scores = []
    for item in items:
        if (
            item["hyperparams"]["task"] != task
            or item["hyperparams"]["reg_type"] != reg_type
        ):
            continue
        reg_strength = item["hyperparams"]["reg_strength"]
        single_score = max(item["results"][results_key])
        reg_strengths.append(reg_strength)
        single_scores.append(single_score)

    return {
        task: {reg_type: {"reg_strength": reg_strengths, "single_score": single_scores}}
    }


def something(json_file, results_key):
    # Assumes everything is the same model.
    items = _load_json(json_file)

    reg_types = set(item["hyperparams"]["reg_type"] for item in items)
    tasks = set(item["hyperparams"]["task"] for item in items)

    ret = collections.defaultdict(dict)

    for task in tasks:
        for reg_type in reg_types:
            summary = get_regularization_effect_on_phase_i_score_summary(
                json_file, results_key=results_key, task=task, reg_type=reg_type
            )
            ret[task].update(summary[task])
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
    json_file_path = BERT_BASE_REGS_MERGE_INFORMED_PAIRS_JSON
    summary = get_merging_single_task_summary(json_file_path)
    s = json.dumps(summary, indent=2)
    print(s)

    ###########################################################################

    # json_file_path = BERT_BASE_REGS_JSON

    # summary = something(
    #     json_file_path, results_key="validation")
    # s = json.dumps(summary, indent=2)
    # print(s)
    # t = reg_effect_summary_to_md_table(summary, reg_type='iso')
    # print(t)
    # print('\n')
    # t = reg_effect_summary_to_md_table(summary, reg_type='ewc')
    # print(t)

    ###########################################################################

    # json_file_path = BERT_BASE_EWC_PHASE_I_JSON

    # train_summary = get_regularization_effect_on_phase_i_score_summary(
    #     json_file_path, results_key="train")
    # validation_summary = get_regularization_effect_on_phase_i_score_summary(
    #     json_file_path, results_key="validation")

    # summary = {
    #     'train': train_summary['sst2'],
    #     'dev': validation_summary['sst2'],
    # }

    # s = json.dumps(summary, indent=2)
    # print(s)
    # t = reg_effect_summary_to_md_table_flipped(summary, reg_type='ewc')
    # print(t)

    ###########################################################################

    # json_file_path = ROBERTA_LARGE_PRELIM_JSON
    # # summary = get_regularization_effect_on_original_score_summary(json_file_path)
    # summary = get_regularization_effect_on_rel_merge_performance_summary(json_file_path, 'iso')
    # s = json.dumps(summary, indent=2)
    # print(s)
    # t = reg_effect_summary_to_md_table(summary, reg_type='iso')
    # print(t)

    ###########################################################################

    # json_file_path = ROBERTA_LARGE_PRELIM_JSON
    # summary = get_merging_single_task_summary(json_file_path)
    # average_improvement = sum(
    #     v.get("absolute_difference", 0.0) for v in summary.values()
    # ) / len(summary.values())

    # summary = {
    #     "summary": summary,
    #     "average_improvement": average_improvement,
    # }

    # s = json.dumps(summary, indent=2)
    # print(s)
