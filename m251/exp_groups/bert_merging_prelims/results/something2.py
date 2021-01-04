"""TODO: Reorganize and rename.


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/something2.py

"""
import os
import json

from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.evaluation import eval_execs
from del8.executables.training import fitting

from m251.fisher.execs import merging_execs


###############################################################################


def _get_single_score(scores):
    if not isinstance(scores, dict):
        return scores
    values = [_get_single_score(v) for v in scores.values()]
    return sum(values) / len(values)


def _get_finished_run_params(exp):
    run_ids = exp.retrieve_run_uuids(RunState.FINISHED)
    run_params = [exp.retrieve_run_params(rid) for rid in run_ids]
    return list(zip(run_ids, run_params))


###############################################################################


@experiment.with_experiment_storages()
def get_eval_results_from_train_run(train_exp, metrics):
    # Only use when trained with with_validation=True.
    run_params = _get_finished_run_params(train_exp)

    items = []
    for run_id, param in run_params:
        history = train_exp.retrieve_single_item_by_class(
            fitting.TrainingHistory, run_id
        )
        history = history.history

        item = {"hyperparams": train_exp.create_run_key_values(param), "results": {}}
        for metric in metrics:
            if isinstance(metrics, dict):
                new_key = metrics[metric]
            else:
                new_key = metric
            if metric in history:
                item["results"][new_key] = [100 * x for x in history[metric]]

        items.append(item)

    return items


###############################################################################


@experiment.with_experiment_storages()
def get_eval_results_from_eval_run(eval_exp):
    # Only use when trained with with_validation=True.
    run_params = _get_finished_run_params(eval_exp)

    items = []
    for run_id, param in run_params:
        eval_results = eval_exp.retrieve_items_by_class(
            eval_execs.CheckpointEvaluationResults, run_id
        )
        ckpt_id_to_results = {r.checkpoint_blob_uuid: r for r in eval_results}

        validation = [
            _get_single_score(ckpt_id_to_results[ckpt_id].results)
            for ckpt_id in param.checkpoints_summary.checkpoint_uuids
        ]

        items.append(
            {
                "hyperparams": eval_exp.create_run_key_values(param),
                "results": {"validation": validation},
            }
        )

    return items


###############################################################################
###############################################################################


if __name__ == "__main__":
    from m251.exp_groups.bert_merging_prelims.exps import finetune_bert_base

    items = get_eval_results_from_eval_run(finetune_bert_base.GlueEval_Regs)

    s = json.dumps(items, indent=2)
    print(s)

    ###########################################################################

    # from m251.exp_groups.bert_merging_prelims.exps import finetune_bert_base

    # items = get_eval_results_from_train_run(
    #     train_exp=finetune_bert_base.GlueEwc_PhaseI,
    #     metrics={'sst2_acc': 'train', 'val_sst2_acc': 'validation'})

    # s = json.dumps(items, indent=2)
    # print(s)
