"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/intermediate_hf/results/multi_merge_results.py

"""
import collections
import csv
import json
import os

import numpy as np

from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES
from m251.fisher.execs import merging_execs
from m251.exp_groups.paper.results import utils as result_utils

from m251.exp_groups.paper.nlp.intermediate_hf import defs

get_single_score = result_utils.get_single_score
result_file = result_utils.result_file


def create_json(merge_exp):
    with merge_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[merge_exp.uuid])

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=merge_exp.uuid)

    items = []
    for run_id in merge_run_ids:
        merge_run = exps_data.get_run_data(run_id)

        params = merge_run.get_single_item_by_class(merge_exp.params_cls)
        reses = merge_run.get_items_by_class(merging_execs.MergingEvaluationResults)

        # print([(r.weighting[0], get_single_score(r.results)) for r in reses])

        res = max(reses, key=lambda r: get_single_score(r.results))
        target_mtm, *donor_mtms = params.models_to_merge

        items.append(
            {
                "target_task": target_mtm.task,
                "donor_tasks": [m.task for m in donor_mtms],
                "trial_index": params.trial_index,
                "chunk_index": params.chunk_index,
                "merged_score": res.results,
                "weighting": res.weighting,
                "target_ckpt": target_mtm.model_checkpoint_uuid,
            }
        )

    return items


def process_json(jason):
    groups = collections.defaultdict(list)
    for item in jason:
        key = (item["trial_index"], tuple(item["donor_tasks"]))
        groups[key].append(item)

    groups = {
        k: max(v, key=lambda x: get_single_score(x["merged_score"]))
        for k, v in groups.items()
    }

    groups2 = collections.defaultdict(list)
    for v in groups.values():
        groups2[tuple(v["donor_tasks"])].append(v)

    for k, v in groups2.items():
        scores = [get_single_score(w["merged_score"]) for w in v]
        print([(w["weighting"]) for w in v])
        avg_score = sum(scores) / len(scores)
        print(f"{k[0]}, {k[1]}: {avg_score:02f}, {len(scores)}")
        # print(scores)


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.intermediate_hf import multi_merge

    # exp = multi_merge.Merge_BertBase_Rte_MnliQnli
    exp = multi_merge.Merge_BertBase_Rte_MnliRte

    jason = create_json(exp)
    process_json(jason)
