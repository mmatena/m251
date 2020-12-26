"""TODO: Reorganize and rename."


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/something.py

"""
from del8.core.storage.storage import RunState

from del8.executables.evaluation import eval_execs

from m251.fisher.execs import merging_execs


def something_evals(eval_exp):
    with eval_exp.get_storage():
        return _something_evals(eval_exp)


def _something_evals(eval_exp):
    # train_run_ids = train_exp.retrieve_run_uuids(RunState.FINISHED)
    # train_run_params = [train_exp.retrieve_run_params(rid) for rid in train_run_ids]

    eval_run_ids = eval_exp.retrieve_run_uuids(RunState.FINISHED)
    eval_run_params = [eval_exp.retrieve_run_params(rid) for rid in eval_run_ids]

    for eval_run_id, p in zip(eval_run_ids, eval_run_params):
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
    merge_run_ids = merge_exp.retrieve_run_uuids(RunState.FINISHED)
    merge_run_params = [merge_exp.retrieve_run_params(rid) for rid in merge_run_ids]

    for merge_run_id, p in zip(merge_run_ids, merge_run_params):
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
###############################################################################

# from m251.exp_groups.bert_merging_prelims.exps import diag_merge_glue_iso
# something_merges(diag_merge_glue_iso.DiagMergeGlueIsoExperiment_LastCkpt_Pairs_Base)
# something_merges(diag_merge_glue_iso.DiagMergeGlueIsoExperiment_LastCkpt_Pairs_Large)

# from m251.exp_groups.bert_merging_prelims.exps import finetune_glue_iso
# something_evals(finetune_glue_iso.FinetuneGlueIsoEval_Base)
# something_evals(finetune_glue_iso.FinetuneGlueIsoEval_Large)
