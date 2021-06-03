"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/intermediate_hf/results/eval_results.py

"""
import collections
import csv
import json
import os

import numpy as np

from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.evaluation import eval_execs
from del8.executables.models import checkpoints

from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES
from m251.fisher.execs import merging_execs
from m251.exp_groups.paper.results import utils as result_utils
from m251.exp_groups.paper.nlp.intermediate_hf import defs

get_single_score = result_utils.get_single_score
result_file = result_utils.result_file


BAD_FINETUNE_RUN_UUIDS = frozenset(
    {
        "37dbf11090b047b2ba2e9996597e22ab",
        "ab6ce15a17ad4ea287c08093270ee494",
        "b8103c8e19054604a420b1ec2c1e4a15",
        "2b23839254934890acd9fab09803382c",
    }
)


def create_json(eval_exp):
    with eval_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[eval_exp.uuid])

    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=eval_exp.uuid)

    items = []
    for run_id in run_ids:
        eval_run = exps_data.get_run_data(run_id)

        params = eval_run.get_single_item_by_class(eval_exp.params_cls)
        res = eval_run.get_single_item_by_class(eval_execs.CheckpointEvaluationResults)
        items.append(
            {
                "task": params.task,
                "trial_index": params.trial_index,
                "score": res.results,
            }
        )

    return items


def _create_ckpt_to_train_params_map(exps_data, train_exp):
    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=train_exp.uuid)

    ret = {}
    for run_id in run_ids:
        run = exps_data.get_run_data(run_id)
        params = run.get_single_item_by_class(train_exp.params_cls)
        ckpt_summary = run.get_single_item_by_class(checkpoints.CheckpointsSummary)
        for ckpt in ckpt_summary.checkpoint_uuids:
            ret[ckpt] = params
    return ret


def create_json_best(eval_exp, sft_exp, train_exp):
    with eval_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(
            experiment_uuid=[eval_exp.uuid, train_exp.uuid, sft_exp.uuid]
        )

    run_ids = exps_data.get_finished_runs_ids(experiment_uuid=eval_exp.uuid)

    ckpt_to_sft_params = _create_ckpt_to_train_params_map(exps_data, sft_exp)
    ckpt_to_train_params = _create_ckpt_to_train_params_map(exps_data, train_exp)

    pt_ckpt_to_task = {v: k for k, v in defs.TASK_TO_CKPT_BERT_BASE.items()}

    items = []
    for run_id in run_ids:
        eval_run = exps_data.get_run_data(run_id)

        eval_params = eval_run.get_single_item_by_class(eval_exp.params_cls)

        res = eval_run.get_items_by_class(eval_execs.CheckpointEvaluationResults)
        best = max(res, key=lambda r: get_single_score(r.results))

        sft_params = ckpt_to_sft_params[best.checkpoint_blob_uuid]

        if sft_params.checkpoint:
            train_params = ckpt_to_train_params[sft_params.checkpoint]
            if train_params.uuid in BAD_FINETUNE_RUN_UUIDS:
                continue
            int_task = train_params.task
        else:
            int_task = pt_ckpt_to_task[sft_params.pretrained_model]

        items.append(
            {
                "task": eval_params.task,
                "intermediate_task": int_task,
                "trial_index": eval_params.trial_index,
                "score": best.results,
            }
        )

    return items


def create_csv_table(items, round_digits=1):

    row_groups = collections.defaultdict(list)
    for item in items:
        group_key = (item["task"], item["intermediate_task"])
        row_groups[group_key].append(item)

    header = [
        "task",
        "int task",
        "score",
        "stddev",
        "num trials",
    ]
    body = []
    for (task, int_task), row_items in row_groups.items():
        scores = [get_single_score(item["score"]) for item in row_items]
        # scores = [s for s in scores if s > 60.0]
        scores = np.array(scores)
        print(scores)
        row = [
            task,
            int_task,
            round(np.mean(scores), round_digits),
            round(np.std(scores), round_digits),
            len(scores),
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
    jasons,
    render_score_fn=latex_render_score_subscript,
    target_task_order=result_utils.GLUE_TASKS_ORDER,
    donor_task_order=result_utils.GLUE_TASKS_ORDER,
    no_original_scores=False,
):
    items = []
    for jason in jasons:
        items.extend(jason)

    row_groups = collections.defaultdict(list)
    for item in items:
        group_key = hashabledict(
            {
                "target_task": item["task"],
                "donor_task": item["intermediate_task"],
            }
        )
        row_groups[group_key].append(item)

    def create_donor_to_merge_summary(target_task):
        ret = {}
        for k, v in row_groups.items():
            if k["target_task"] != target_task:
                continue
            ret[k["donor_task"]] = v

        ret2 = {}
        for donor_task, ret_items in ret.items():
            merged_scores = np.array(
                [get_single_score(item["score"]) for item in ret_items]
            )
            mean = np.mean(merged_scores)
            stddev = np.std(merged_scores) if len(ret_items) > 1 else None
            ret2[donor_task] = (mean, stddev)
        return ret2

    def get_original_task_summary(task):
        ret = {}
        for k, v in row_groups.items():
            if k["target_task"] != target_task:
                continue
            ret[k["donor_task"]] = v

        if not ret:
            return None, None

        ret_items = max(ret.values(), key=len)
        merged_scores = np.array(
            [get_single_score(item["original_score"]) for item in ret_items]
        )
        mean = np.mean(merged_scores)
        stddev = np.std(merged_scores) if len(ret_items) > 1 else None

        return mean, stddev

    rows = [len(target_task_order) * [""] for _ in donor_task_order]

    for col_idx, target_task in enumerate(target_task_order):
        donor_to_merge_summary = create_donor_to_merge_summary(target_task)

        for row_idx, donor_task in enumerate(donor_task_order):
            if donor_task == target_task and not no_original_scores:
                mean, stddev = get_original_task_summary(target_task)
                rows[row_idx][col_idx] = render_score_fn(mean, stddev, is_orig=True)
                continue

            if donor_task not in donor_to_merge_summary:
                continue
            mean, stddev = donor_to_merge_summary[donor_task]
            rows[row_idx][col_idx] = render_score_fn(mean, stddev)

    for row, task in zip(rows, donor_task_order):
        row.insert(0, result_utils.TASK_NICE_NAMES[task])

    rows = [
        R"\toprule",
        [R"\textbf{Task}"]
        + [result_utils.TASK_NICE_NAMES[t] for t in target_task_order],
        R"\midrule",
        *rows,
        R"\bottomrule",
    ]

    return result_utils.table_to_latex(rows)


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.intermediate_hf import eval as evl
    from m251.exp_groups.paper.nlp.intermediate_hf import eval_large
    from m251.exp_groups.paper.nlp.intermediate_hf import sft
    from m251.exp_groups.paper.nlp.intermediate_hf import sft_large
    from m251.exp_groups.paper.nlp.intermediate_hf import finetune
    from m251.exp_groups.paper.nlp.intermediate import finetune as og_ft

    ###########################################################################

    # eval_exp = evl.Eval_LrSrc_Sft_LastCkpt
    # eval_exp = evl.Eval_HrSrc_Sft_LastCkpt
    eval_exp = evl.EvalBertBaseFromMnli_Sft_LastCkpt
    # eval_exp = eval_large.Eval_Large_Sft_LastCkpt

    # sft_exp = sft.GlueFinetune_BertBase_LrSrc_Sft
    # sft_exp = sft.GlueFinetune_BertBase_HrSrc_Sft
    sft_exp = sft.GlueFinetune_BertBaseFromMnli_Sft
    # sft_exp = sft_large.Large_Sft

    # train_exp = finetune.GlueFinetune_BertBase
    train_exp = finetune.GlueFinetune_BertBaseFromMnliCkpt
    # train_exp = og_ft.GlueFinetune

    summary = create_json_best(eval_exp, sft_exp, train_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)

    # t = create_csv_table(summary)
    t = create_latex_table(
        [
            # create_json_best(evl.Eval_LrSrc_Sft_LastCkpt, sft.GlueFinetune_BertBase_LrSrc_Sft, train_exp),
            # create_json_best(evl.Eval_HrSrc_Sft_LastCkpt, sft.GlueFinetune_BertBase_HrSrc_Sft, train_exp),
            summary,
        ],
        # target_task_order=('cola', 'mrpc', 'stsb', 'rte'),
        target_task_order=("mrpc", "stsb", "rte"),
        donor_task_order=("mrpc", "stsb", "rte"),
        no_original_scores=True,
    )
    print(t)
