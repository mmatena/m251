"""TODO: Reorganize and rename.

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


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/generate_merging_json.py

python3 -i m251/exp_groups/bert_merging_prelims/results/generate_merging_json.py

"""
import os
import json

from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from m251.fisher.execs import merging_execs

from m251.storage_util import eval_results

###############################################################################


def _to_mnli(task):
    if "mnli" in task:
        task = "mnli"
    return task


def _extract_common_params(run_params):
    return hashabledict(
        pretrained_model=run_params.pretrained_model,
        reg_strength=run_params.reg_strength,
        reg_type=run_params.reg_type,
    )


def generate_merging_json_for_best_ckpt(train_exp, eval_exp, merge_exp):
    with train_exp.get_storage(), eval_exp.get_storage(), merge_exp.get_storage():
        return _generate_merging_json_for_best_ckpt(train_exp, eval_exp, merge_exp)


def _generate_merging_json_for_best_ckpt(train_exp, eval_exp, merge_exp):
    ret = []

    # Eval stuff.
    ckpt_to_group_and_results = {}
    eval_run_ids = eval_exp.retrieve_run_uuids(RunState.FINISHED)
    for run_id in eval_run_ids:
        run_params = eval_exp.retrieve_run_params(run_id)

        common_params = _extract_common_params(run_params)

        best_result = eval_results.get_best_result_for_run(
            eval_exp, run_id, run_params.task
        )

        # proper_task = _to_mnli(run_params.task)

        ckpt = best_result.checkpoint_blob_uuid
        existing = ckpt_to_group_and_results.get(ckpt, None)
        if existing:
            assert existing[0] == common_params
            existing[1].update(best_result)
        else:
            ckpt_to_group_and_results[ckpt] = (common_params, best_result)

    # Merging stuff.
    merging_run_ids = merge_exp.retrieve_run_uuids(RunState.FINISHED)
    for run_id in merging_run_ids:
        merge_results = merge_exp.retrieve_items_by_class(
            merging_execs.MergingEvaluationResults, run_id
        )
        for merge_result in merge_results:
            iter_ = zip(
                merge_result.tasks, merge_result.checkpoints, merge_result.weighting
            )
            for task, checkpoint, weight in iter_:
                # merge_result.results
                # original_result.results
                common_params, original_result = ckpt_to_group_and_results[checkpoint]

                if task == "mnli":
                    original_score = original_result.results
                    merged_score = {
                        k: v for k, v in merge_result.results.items() if "mnli" in k
                    }
                else:
                    original_score = original_result.results[task]
                    merged_score = merge_result.results[task]

                item = {
                    "task": task,
                    "other_task": [t for t in merge_result.tasks if t != task][0],
                    "hyperparams": common_params,
                    "original_score": original_score,
                    "merged_score": merged_score,
                    "weighting": weight,
                }
                ret.append(item)

    return ret


###############################################################################
###############################################################################


if True:
    from m251.exp_groups.bert_merging_prelims.exps import finetune_glue_iso
    from m251.exp_groups.bert_merging_prelims.exps import diag_merge_glue_iso

    result = generate_merging_json_for_best_ckpt(
        train_exp=finetune_glue_iso.FinetuneGlueIsoExperiment_RobertaLarge,
        eval_exp=finetune_glue_iso.FinetuneGlueIsoRobustEval_RobertaLarge,
        merge_exp=diag_merge_glue_iso.DiagMergeGlueIsoExperiment_BestCkpt_Pairs_RobertaLarge,
    )

    json_path = "~/Desktop/projects/m251/local_results/roberta_large_prelim.json"
    json_path = os.path.expanduser(json_path)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
