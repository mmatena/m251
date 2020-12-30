"""TODO: Add title."""
import collections
import re

from del8.core.experiment import runs
from del8.core.storage.storage import RunState

from del8.executables.evaluation import eval_execs


def retrieve_all_for_run(eval_exp, run_uuid):
    return eval_exp.retrieve_items_by_class(
        eval_execs.CheckpointEvaluationResults, run_uuid
    )


def retrieve_combined_results_for_runs(eval_exp, run_uuids):
    ckpt_to_results = collections.defaultdict(list)

    for run_uuid in run_uuids:
        eval_results = retrieve_all_for_run(eval_exp, run_uuid)
        for result in eval_results:
            ckpt_to_results[result.checkpoint_blob_uuid].append(result.results)

    eval_results = []
    for ckpt, results in ckpt_to_results.items():
        combined_results = {}
        for r in results:
            combined_results.update(r)
        eval_results.append(
            eval_execs.CheckpointEvaluationResults(
                results=combined_results, checkpoint_blob_uuid=ckpt
            )
        )

    return eval_results


# def task_score_from_results(results, task, average=False):
#     if hasattr(results, 'results'):
#         # Let us pass in a result @data_class or the raw
#         # results dict.
#         results = results.results

#     task_items = {
#         k: v
#         for k, v in results.items()
#         # NOTE: This is a first pass regex. It will get complicated as
#         # time goes on.
#         if re.search(rf'(?:^|_){task}(?:_|$)', k)
#     }
#     print(task_items)

#     if not task_items:
#         raise ValueError(f'No scores for task {task} found in results {results}.')
#     elif len(task_items) > 1 and not average:
#         raise ValueError(
#             f'Found multiples scores with keys {task_items.keys()} for '
#             'task {task} in results {results}.')
#     return sum(task_items.values()) / len(task_items.values())


def _average_dict_values(d):
    values = d.values()
    return sum(values) / len(values)


def task_score_from_results(results, task):
    if hasattr(results, "results"):
        # Let us pass in a result @data_class or the raw
        # results dict.
        results = results.results

    if task == "mnli":
        scores = [
            results.get("mnli", None),
            results.get("mnli_mismatched", None),
            results.get("mnli_mismatched", None),
        ]
        scores = [s for s in scores if s is not None]
        assert scores, "No keys found for mnli."
    else:
        scores = [results[task]]

    scores = [_average_dict_values(s) for s in scores]

    return sum(scores) / len(scores)


def task_scores_from_results(results, tasks, average=False):
    if isinstance(tasks, str):
        raise ValueError(
            "The function task_scores_from_results tasks a list of tuple as its task "
            f"argument the string {tasks} was passed instead."
        )
    return {
        task: task_score_from_results(results, task, average=average) for task in tasks
    }


def get_best_result_for_run(eval_exp, run_uuid, task):
    # Best means highest validation score.
    if task == "mnli" and isinstance(run_uuid, (list, tuple)):
        results = retrieve_combined_results_for_runs(eval_exp, run_uuid)
    else:
        results = retrieve_all_for_run(eval_exp, run_uuid)
    return max(results, key=lambda res: task_score_from_results(res, task))


def _remove_task_from_run_key(run_key):
    task = run_key.key_values["task"]

    new_key_values = run_key.key_values.copy()
    del new_key_values["task"]

    new_run_key = run_key.copy(key_values=new_key_values)
    return new_run_key, task


def _do_train_eval_tasks_match(train_task, eval_task):
    if train_task == eval_task:
        return True
    elif train_task == "mnli" and eval_task in ["mnli_matched", "mnli_mismatched"]:
        return True
    else:
        return False


def _do_train_eval_run_keys_match(train_run_key, eval_run_key):
    train_run_key, train_task = _remove_task_from_run_key(train_run_key)
    eval_run_key, eval_task = _remove_task_from_run_key(eval_run_key)
    return _do_train_eval_tasks_match(
        train_task, eval_task
    ) and runs.RunKey.has_same_values(train_run_key, eval_run_key)


def get_eval_run_uuid_for_train_run(train_exp, eval_exp, train_run_uuid):
    train_run_key = train_exp.retrieve_run_key(train_run_uuid)
    train_task = train_run_key.key_values["task"]

    eval_run_uuids = eval_exp.retrieve_run_uuids(RunState.FINISHED)
    matching_eval_run_uuids = []
    for eval_run_uuid in eval_run_uuids:
        eval_run_key = eval_exp.retrieve_run_key(eval_run_uuid)
        if _do_train_eval_run_keys_match(train_run_key, eval_run_key):
            matching_eval_run_uuids.append(eval_run_uuid)

    if not matching_eval_run_uuids:
        raise ValueError(
            f"No matching evaluation run found for train run {train_run_uuid}."
        )
    elif len(matching_eval_run_uuids) > 2 or (
        len(matching_eval_run_uuids) == 2 and train_task != "mnli"
    ):
        raise ValueError(
            f"Found {len(matching_eval_run_uuids)} matching evaluation runs found "
            f"for train run {train_run_uuid} with {train_task} with uuids {set(matching_eval_run_uuids)}."
        )

    if len(matching_eval_run_uuids) == 1:
        return matching_eval_run_uuids[0]
    else:
        return matching_eval_run_uuids
