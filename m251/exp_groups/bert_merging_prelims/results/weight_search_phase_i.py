"""TODO: Something


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/weight_search_phase_i.py

"""
import json

from del8.core import serialization
from del8.core.storage.storage import RunState

from m251.fisher.execs import merging_execs

from m251.exp_groups.bert_merging_prelims.exps import merge_bert_base
from m251.exp_groups.bert_merging_prelims.exps import finetune_bert_base


# EXP = merge_bert_base.MergeWeightSearch_PhaseI
# EXP = merge_bert_base.MergeWeightSearch_GlueRegs
EXP = merge_bert_base.MergeWeightSearch_GP_PhaseI

TREXP = finetune_bert_base.Glue_Regs

with EXP.get_storage(), TREXP.get_storage():
    for run_id in EXP.retrieve_run_uuids(RunState.FINISHED):
        res = EXP.retrieve_single_item_by_class(
            merging_execs.MergingEvaluationResults, run_id
        )

        params = EXP.retrieve_run_params(run_id)
        mtm = params.models_to_merge[0]
        train_params = TREXP.retrieve_run_params(mtm.train_run_uuid)

        # NOTE: res.tasks was incorrectly bound! It is only rte here.
        weighting = {
            mtm.task: w / sum(res.weighting)
            for w, mtm in zip(res.weighting, params.models_to_merge)
        }

        print(TREXP.create_run_key_values(train_params))
        print(weighting)
        print(res.results)
        print("\n")
