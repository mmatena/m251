"""TODO: Something


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/merge_pairs_informed.py


JSON schema:
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
"""
import collections
import json
import os

from del8.core import serialization
from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.evaluation import eval_execs

from m251.fisher.execs import merging_execs

from m251.exp_groups.bert_merging_prelims.exps import merge_bert_base_informed
from m251.exp_groups.bert_merging_prelims.exps import merge_cross_regs
from m251.exp_groups.bert_merging_prelims.exps import finetune_bert_base


BERT_BASE_REGS_MERGE_INFORMED_PAIRS_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_regs_merge_informed_pairs.json"
)

BERT_BASE_MERGE_CROSS_REGS_MNLI_0003_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_merge_cross_regs_mnli_0003.json"
)


def _load_json(json_file):
    if not isinstance(json_file, str):
        # Assume this is the actual json object.
        return json_file
    json_file = os.path.expanduser(json_file)
    with open(json_file, "r") as f:
        return json.load(f)


def _get_single_score(scores):
    if not isinstance(scores, dict):
        return scores
    values = [_get_single_score(v) for v in scores.values()]
    return sum(values) / len(values)


@experiment.with_experiment_storages()
def create_json(merge_exp, train_exp, eval_exp):
    og_evals = eval_exp.retrieve_items_by_class(
        eval_execs.CheckpointEvaluationResults, run_uuid=None
    )
    items = []
    for run_id in merge_exp.retrieve_run_uuids(RunState.FINISHED):
        params = merge_exp.retrieve_run_params(run_id)

        reses = merge_exp.retrieve_items_by_class(
            merging_execs.MergingEvaluationResults, run_id
        )
        res = max(reses, key=lambda r: _get_single_score(r.results))

        for eval_res in og_evals:
            if (
                eval_res.checkpoint_blob_uuid
                == params.models_to_merge[0].model_checkpoint_uuid
            ):
                break

        # train_params = train_exp.retrieve_run_params(params.models_to_merge[0].train_run_uuid)

        items.append(
            {
                "task": params.models_to_merge[0].task,
                "other_task": params.models_to_merge[1].task,
                "hyperparams": {
                    "pretrained_model": params.pretrained_model,
                    "reg_strength": params.reg_strength,
                    "reg_type": params.reg_type,
                    # 'pretrained_model': train_params.pretrained_model,
                    # 'reg_strength': train_params.reg_strength,
                    # 'reg_type': train_params.reg_type,
                },
                "original_score": eval_res.results,
                "merged_score": res.results,
                "weighting": res.weighting[0],
            }
        )

    return items


def json_to_best_per_target(filepath):
    items = _load_json(filepath)

    hp_task_to_items = collections.defaultdict(list)
    for item in items:
        hp_task = {"task": item["task"]}
        hp_task.update(item["hyperparams"])
        hp_task = hashabledict(hp_task)
        hp_task_to_items[hp_task].append(item)

    ret = []
    for task_items in hp_task_to_items.values():
        best_item = max(task_items, key=lambda r: _get_single_score(r["merged_score"]))
        ret.append(best_item)

    return ret


def json_to_merge_score_md_table(filepath):
    items = json_to_best_per_target(filepath)

    tasks = set()
    reg_types = set()
    reg_strengths = set()
    for item in items:
        tasks.add(item["task"])
        reg_types.add(item["hyperparams"]["reg_type"])
        reg_strengths.add(item["hyperparams"]["reg_strength"])

    tasks = sorted(tasks)
    reg_strengths = sorted(reg_strengths)

    tables = []
    for reg_type in reg_types:
        rows = [
            "|" + ("|".join([f"*{reg_type}*"] + tasks)) + "|",
            "|" + ((len(tasks) + 1) * "---|"),
        ]
        for reg_strength in reg_strengths:
            row = [f"**_{reg_strength}_**"]
            for task in tasks:
                for item in items:
                    a = item["task"] == task
                    b = item["hyperparams"]["reg_type"] == reg_type
                    c = item["hyperparams"]["reg_strength"] == reg_strength
                    if a and b and c:
                        # score = _get_single_score(item['merged_score'])
                        #
                        score = _get_single_score(
                            item["merged_score"]
                        ) - _get_single_score(item["original_score"])

                        score = str(round(score, 1))
                        row.append(score)
            row = "|".join(row)
            row = f"|{row}|"
            rows.append(row)
        tables.append("\n".join(rows))

    return "\n\n".join(tables)


if True:
    filepath = BERT_BASE_MERGE_CROSS_REGS_MNLI_0003_JSON
    t = json_to_merge_score_md_table(filepath)
    print(t)

    ###########################################################################

    # merge_exp = merge_cross_regs.MergeRteMrpcWith0003Mnli
    # eval_exp = finetune_bert_base.GlueEval_Regs
    # train_exp = finetune_bert_base.Glue_Regs
    # summary = create_json(merge_exp, train_exp, eval_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)

    ###########################################################################

    # filepath = BERT_BASE_REGS_MERGE_INFORMED_PAIRS_JSON
    # t = json_to_merge_score_md_table(filepath)
    # print(t)

    ###########################################################################

    # merge_exp = merge_bert_base_informed.MergePairsInformed_GlueRegs
    # eval_exp = finetune_bert_base.GlueEval_Regs
    # train_exp = finetune_bert_base.Glue_Regs
    # summary = create_json(merge_exp, train_exp, eval_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)
