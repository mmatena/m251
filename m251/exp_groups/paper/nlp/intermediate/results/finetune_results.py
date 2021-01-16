"""
export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/paper/nlp/intermediate/results/finetune_results.py

"""
import collections
import csv
import json
import os

import numpy as np

from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict
from del8.executables.training import fitting

from m251.data.processing.constants import NUM_GLUE_TRAIN_EXAMPLES
from m251.fisher.execs import merging_execs
from m251.exp_groups.paper.results import utils as result_utils

get_single_score = result_utils.get_single_score
result_file = result_utils.result_file


def _nice(phlote):
    return str(round(phlote, 1))


def examine(train_exp):
    with train_exp.get_storage() as storage:
        exps_data = storage.retrieve_storage_data(experiment_uuid=[train_exp.uuid])

    merge_run_ids = exps_data.get_finished_runs_ids(experiment_uuid=train_exp.uuid)

    out = []

    for run_id in merge_run_ids:
        merge_run = exps_data.get_run_data(run_id)

        params = merge_run.get_single_item_by_class(train_exp.params_cls)
        history = merge_run.get_single_item_by_class(fitting.TrainingHistory)
        history = history.history

        accs = history[f"{params.task}_acc"]

        # final_acc = accs[-1]
        # max_acc = max(accs)
        # out.append(f"{params.task}: {_nice(final_acc)}, {_nice(max_acc)}")

        if params.task in {"qqp", "cola", "mrpc"}:
            accs_str = ", ".join(_nice(a) for a in accs)
            out.append(f"{params.task}: {accs_str}")

    out = "\n".join(sorted(out))
    print(out)


if __name__ == "__main__":
    from m251.exp_groups.paper.nlp.intermediate import finetune

    ###########################################################################

    merge_exp = finetune.GlueFinetune
    examine(merge_exp)
