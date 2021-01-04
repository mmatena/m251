"""TODO: Something


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/merge_single_target_all_donor_ckpts.py

"""
import collections
import json
import os

from del8.core import serialization
from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.evaluation import eval_execs
from del8.executables.models import checkpoints

from m251.fisher.execs import merging_execs

from m251.exp_groups.bert_merging_prelims.exps import bert_base_all_mnli_to_rte
from m251.exp_groups.bert_merging_prelims.exps import bert_base_all_mnli_to_mrpc
from m251.exp_groups.bert_merging_prelims.exps import finetune_bert_base


BERT_BASE_MERGE_RTE_ALL_MNLI_CKPTS_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_merge_rte_all_mnli_ckpts.json"
)

BERT_BASE_MERGE_MRPC_ALL_MNLI_CKPTS_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_merge_mrpc_all_mnli_ckpts.json"
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
    ckpt_summaries = train_exp.retrieve_items_by_class(
        checkpoints.CheckpointsSummary, run_uuid=None
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

        for ckpt_summary in ckpt_summaries:
            if (
                params.models_to_merge[1].model_checkpoint_uuid
                in ckpt_summary.checkpoint_uuids
            ):
                ckpt_index = ckpt_summary.checkpoint_uuids.index(
                    params.models_to_merge[1].model_checkpoint_uuid
                )
                break

        items.append(
            {
                "task": params.models_to_merge[0].task,
                "donor_task": params.models_to_merge[1].task,
                "donor_ckpt_index": ckpt_index,
                "hyperparams": {
                    "pretrained_model": params.pretrained_model,
                    "reg_strength": params.reg_strength,
                    "reg_type": params.reg_type,
                },
                "original_score": eval_res.results,
                "merged_score": res.results,
                "weighting": res.weighting[0],
            }
        )

    return items


def json_to_merge_score_md_table(filepath, target_task):
    items = _load_json(filepath)

    ckpt_indices = set()
    reg_types = set()
    reg_strengths = set()
    for item in items:
        ckpt_indices.add(item["donor_ckpt_index"])
        reg_types.add(item["hyperparams"]["reg_type"])
        reg_strengths.add(item["hyperparams"]["reg_strength"])

    ckpt_indices = sorted(ckpt_indices)
    reg_strengths = sorted(reg_strengths)

    tables = []
    for reg_type in reg_types:
        rows = [
            "|" + ("|".join([f"*{reg_type}*"] + [str(c) for c in ckpt_indices])) + "|",
            "|" + ((len(ckpt_indices) + 1) * "---|"),
        ]
        for reg_strength in reg_strengths:
            row = [f"**_{reg_strength}_**"]
            for ckpt_index in ckpt_indices:
                for item in items:
                    a = item["donor_ckpt_index"] == ckpt_index
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
    filepath = BERT_BASE_MERGE_MRPC_ALL_MNLI_CKPTS_JSON
    t = json_to_merge_score_md_table(filepath, target_task="mrpc")
    print(t)

    ###########################################################################

    # merge_exp = bert_base_all_mnli_to_mrpc.MergeBestMrpcWithEachMnli_GlueRegs
    # eval_exp = finetune_bert_base.GlueEval_Regs
    # train_exp = finetune_bert_base.Glue_Regs
    # summary = create_json(merge_exp, train_exp, eval_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)

    ###########################################################################

    # filepath = BERT_BASE_MERGE_RTE_ALL_MNLI_CKPTS_JSON
    # t = json_to_merge_score_md_table(filepath, target_task='rte')
    # print(t)

    ###########################################################################

    # merge_exp = bert_base_all_mnli_to_rte.MergeBestRteWithEachMnli_GlueRegs
    # eval_exp = finetune_bert_base.GlueEval_Regs
    # train_exp = finetune_bert_base.Glue_Regs
    # summary = create_json(merge_exp, train_exp, eval_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)
