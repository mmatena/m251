"""TODO: Reorganize and rename.


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/something.py

"""
import os
import json

from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.evaluation import eval_execs

from m251.fisher.execs import merging_execs


###############################################################################


def _get_finished_run_params(exp):
    run_ids = exp.retrieve_run_uuids(RunState.FINISHED)
    run_params = [exp.retrieve_run_params(rid) for rid in run_ids]
    return list(zip(run_ids, run_params))


###############################################################################


def something_evals(eval_exp):
    with eval_exp.get_storage():
        return _something_evals(eval_exp)


def _something_evals(eval_exp):
    eval_run_params = _get_finished_run_params(eval_exp)

    for eval_run_id, p in eval_run_params:
        checkpoints_summary = p.checkpoints_summary
        eval_results = eval_exp.retrieve_items_by_class(
            eval_execs.CheckpointEvaluationResults, eval_run_id
        )

        ckpt_id_to_results = {r.checkpoint_blob_uuid: r for r in eval_results}

        print(eval_exp.create_run_key_values(p))

        for ckpt_uuid in checkpoints_summary.checkpoint_uuids:
            results = ckpt_id_to_results[ckpt_uuid]
            print(results.results)

        print(2 * "\n")


###############################################################################


def something_merges(merge_exp):
    with merge_exp.get_storage():
        return _something_merges(merge_exp)


def _something_merges(merge_exp):
    merge_run_params = _get_finished_run_params(merge_exp)

    for merge_run_id, p in merge_run_params:
        merge_results = merge_exp.retrieve_items_by_class(
            merging_execs.MergingEvaluationResults, merge_run_id
        )
        for merge_result in merge_results:
            # results, checkpoints, tasks, weighting
            print(merge_result.tasks)
            print(merge_result.weighting)
            print(merge_result.results)
            print()

        print(2 * "\n")


###############################################################################


def _eval_param_to_key(eval_exp, eval_param):
    return hashabledict(
        {
            "pretrained_model": eval_param.pretrained_model,
            "task": eval_param.task,
            "reg_strength": eval_param.reg_strength,
            "reg_type": eval_param.reg_type,
        }
    )


def _merge_param_to_key(train_exp, merge_exp, merge_param):
    for m in merge_param.models_to_merge:
        train_params = train_exp.retrieve_run_params(m.train_run_uuid)
        return {
            "pretrained_model": train_params.pretrained_model,
            "reg_strength": train_params.reg_strength,
            "reg_type": train_params.reg_type,
        }


def something_joined(train_exp, eval_exp, merge_exp, ckpt_index):
    # TODO: Try to make the ckpt_index selection automatic from experiments/retrieved
    # items. That'll probably prevent silly mistakes.
    with train_exp.get_storage(), eval_exp.get_storage(), merge_exp.get_storage():
        return _something_joined(train_exp, eval_exp, merge_exp, ckpt_index=ckpt_index)


def _key_results_by_task(results, tasks):
    ret = {}
    for task in tasks:
        for k, v in results.items():
            if f"{task}_acc" in k:
                ret[task] = v
                break
    return ret


def _something_joined(train_exp, eval_exp, merge_exp, ckpt_index):
    merge_run_params = _get_finished_run_params(merge_exp)
    eval_run_params = _get_finished_run_params(eval_exp)

    ret = []

    eval_key_to_results = {}
    for eval_id, p in eval_run_params:
        key = _eval_param_to_key(eval_exp, p)
        ckpt = p.checkpoints_summary.checkpoint_uuids[ckpt_index]
        eval_results = eval_exp.retrieve_items_by_class(
            eval_execs.CheckpointEvaluationResults, eval_id
        )
        ckpt_id_to_results = {r.checkpoint_blob_uuid: r for r in eval_results}

        eval_key_to_results[key] = ckpt_id_to_results[ckpt]

    for merge_run_id, p in merge_run_params:
        merge_results = merge_exp.retrieve_items_by_class(
            merging_execs.MergingEvaluationResults, merge_run_id
        )

        task_to_eval_score = {}
        merge_key = _merge_param_to_key(train_exp, merge_exp, p)
        for m in p.models_to_merge:
            eval_key = {"task": m.task}
            eval_key.update(merge_key)
            eval_key = hashabledict(eval_key)
            task_to_eval_score[m.task] = eval_key_to_results[eval_key].results[
                f"{m.task}_acc"
            ]

        ret_item = {
            "common_parameters": merge_key.copy(),
            "original_scores": task_to_eval_score.copy(),
            "merge_results": [],
        }
        for merge_result in merge_results:
            scores = {}
            for task in merge_result.tasks:
                eval_score = task_to_eval_score[task]
                for name, merge_score in merge_result.results.items():
                    if task in name:
                        break
                scores[task] = (merge_score, eval_score)
            print(merge_key)
            print(merge_result.weighting)
            print(scores)
            print()

            merged_scores = _key_results_by_task(
                merge_result.results, merge_result.tasks
            )
            relative_difference = {
                task: merged_score / task_to_eval_score[task]
                for task, merged_score in merged_scores.items()
            }
            absolute_difference = {
                task: 100 * (merged_score - task_to_eval_score[task])
                for task, merged_score in merged_scores.items()
            }
            weighting = {
                task: weight
                for task, weight in zip(merge_result.tasks, merge_result.weighting)
            }
            merged_ret = {
                "weighting": weighting,
                "scores": merged_scores,
                "relative_difference": relative_difference,
                "absolute_difference": absolute_difference,
            }
            ret_item["merge_results"].append(merged_ret)

        ret.append(ret_item)
        print(2 * "\n")
    return ret


###############################################################################
###############################################################################


if True:
    pass
    # from m251.exp_groups.simclr_merging_prelims.exps import simclr_finetune_iso
    # from m251.exp_groups.simclr_merging_prelims.exps import simclr_diag_merge_iso

    from m251.exp_groups.bert_merging_prelims.exps import finetune_glue_iso
    from m251.exp_groups.bert_merging_prelims.exps import diag_merge_glue_iso

    # something_merges(diag_merge_glue_iso.DiagMergeGlueIsoExperiment_LastCkpt_Pairs_Base)
    # something_merges(diag_merge_glue_iso.DiagMergeGlueIsoExperiment_LastCkpt_Pairs_Large)
    # something_merges(diag_merge_glue_iso.DiagMergeGlueIsoExperiment_BestCkpt_Pairs_RobertaLarge)

    # something_evals(finetune_glue_iso.FinetuneGlueIsoEval_Base)
    # something_evals(finetune_glue_iso.FinetuneGlueIsoEval_Large)
    # something_evals(finetune_glue_iso.FinetuneGlueIsoRobustEval_RobertaLarge)

    # something_evals(simclr_finetune_iso.FinetuneSimclrIsoEval_r50_1x)
    # something_merges(simclr_diag_merge_iso.SimclrDiagMergeIso__r50_1x__ckpt_20k)

    ############

    # result = something_joined(
    #     train_exp=simclr_finetune_iso.FinetuneSimclrIso_r50_1x,
    #     eval_exp=simclr_finetune_iso.FinetuneSimclrIsoEval_r50_1x,
    #     merge_exp=simclr_diag_merge_iso.SimclrDiagMergeIso__r50_1x__ckpt_20k,
    #     ckpt_index=1,
    # )
    # s = json.dumps(result, indent=2)
    # print(s)

    ############

    # result = something_joined(
    #     train_exp=finetune_glue_iso.FinetuneGlueIsoExperiment_Large,
    #     eval_exp=finetune_glue_iso.FinetuneGlueIsoEval_Large,
    #     merge_exp=diag_merge_glue_iso.DiagMergeGlueIsoExperiment_LastCkpt_Pairs_Large,
    #     ckpt_index=-1,
    # )
    # # s = json.dumps(result, indent=2)
    # # print(s)
    # json_path = os.path.expanduser('~/Desktop/projects/m251/local_results/bert_prelim_large.json')
    # with open(json_path, "w") as f:
    #     json.dump(result, f, indent=2)
