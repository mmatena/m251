"""TODO: Something


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/merge_all_targets_all_donors.py

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

from m251.exp_groups.bert_merging_prelims.exps import bert_base_merge_pair_all_ckpts
from m251.exp_groups.bert_merging_prelims.exps import finetune_bert_base


BERT_BASE_ALL_MNLI_TO_ALL_RTE_0_0003_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_all_mnli_to_all_rte_0_0003.json"
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
                params.models_to_merge[0].model_checkpoint_uuid
                in ckpt_summary.checkpoint_uuids
            ):
                target_ckpt_index = ckpt_summary.checkpoint_uuids.index(
                    params.models_to_merge[0].model_checkpoint_uuid
                )
            if (
                params.models_to_merge[1].model_checkpoint_uuid
                in ckpt_summary.checkpoint_uuids
            ):
                donor_ckpt_index = ckpt_summary.checkpoint_uuids.index(
                    params.models_to_merge[1].model_checkpoint_uuid
                )

        items.append(
            {
                "task": params.models_to_merge[0].task,
                "target_ckpt_index": target_ckpt_index,
                "donor_task": params.models_to_merge[1].task,
                "donor_ckpt_index": donor_ckpt_index,
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
    # Assumes a single reg_type, donor_task, target task, and reg strength.
    items = _load_json(filepath)

    donor_ckpt_indices = set()
    target_ckpt_indices = set()
    for item in items:
        donor_ckpt_indices.add(item["donor_ckpt_index"])
        target_ckpt_indices.add(item["target_ckpt_index"])

    donor_ckpt_indices = sorted(donor_ckpt_indices)
    target_ckpt_indices = sorted(target_ckpt_indices)

    og_scores = [_get_single_score(item["original_score"]) for item in items]
    best_og_score = max(og_scores)

    rows = [
        "|"
        + ("|".join(["v target \\ donor >"] + [str(c) for c in donor_ckpt_indices]))
        + "|",
        "|" + ((len(donor_ckpt_indices) + 1) * "---|"),
    ]
    for target_ckpt_index in target_ckpt_indices:
        row = [f"**_{target_ckpt_index}_**"]
        for donor_ckpt_index in donor_ckpt_indices:
            for item in items:
                a = item["donor_ckpt_index"] == donor_ckpt_index
                b = item["target_ckpt_index"] == target_ckpt_index
                if a and b:
                    # score = _get_single_score(item['merged_score'])
                    #
                    # score = _get_single_score(item['merged_score']) - _get_single_score(item['original_score'])
                    #
                    score = _get_single_score(item["merged_score"]) - best_og_score

                    score = str(round(score, 1))
                    row.append(score)
        row = "|".join(row)
        row = f"|{row}|"
        rows.append(row)

    return "\n".join(rows)


if True:
    filepath = BERT_BASE_ALL_MNLI_TO_ALL_RTE_0_0003_JSON
    t = json_to_merge_score_md_table(filepath, target_task="mrpc")
    print(t)

    ###########################################################################

    # merge_exp = bert_base_merge_pair_all_ckpts.MergeEachRteWithEachMnli_Iso_0_0003_GlueRegs
    # eval_exp = finetune_bert_base.GlueEval_Regs
    # train_exp = finetune_bert_base.Glue_Regs
    # summary = create_json(merge_exp, train_exp, eval_exp)
    # s = json.dumps(summary, indent=2)
    # print(s)
