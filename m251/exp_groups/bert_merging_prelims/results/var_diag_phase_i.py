"""TODO: Something


export PYTHONPATH=$PYTHONPATH:~/Desktop/projects/m251:~/Desktop/projects/del8

python3 m251/exp_groups/bert_merging_prelims/results/var_diag_phase_i.py

"""
import collections
import itertools
import json
import os

from del8.core import serialization
from del8.core.experiment import experiment
from del8.core.storage.storage import RunState
from del8.core.utils.type_util import hashabledict

from del8.executables.evaluation import eval_execs
from del8.executables.models import checkpoints

from m251.fisher.execs import fisher_execs
from m251.fisher.execs import merging_execs

from m251.exp_groups.bert_merging_prelims.exps import bert_base_fisher_var_diag
from m251.exp_groups.bert_merging_prelims.exps import bert_base_var_diag_merge_phase_i
from m251.exp_groups.bert_merging_prelims.exps import bert_base_var_diag_merge_phase_ii
from m251.exp_groups.bert_merging_prelims.exps import finetune_bert_base


BERT_BASE_VAR_DIAG_RTE_MNLI_0_0003_PHASE_I_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_var_diag_rte_mnli_0_0003_phase_i.json"
)

BERT_BASE_VAR_DIAG_RTE_MNLI_0_0003_PHASE_II_JSON = os.path.expanduser(
    "~/Desktop/projects/m251/local_results/bert_base_var_diag_rte_mnli_0_0003_phase_ii.json"
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


def _tranpose(rows):
    # Discards no data if jagged and fills short nested lists with empty string.
    return list(map(list, itertools.zip_longest(*rows, fillvalue="")))


def _to_md_table(rows):
    header, *body = rows
    header = [
        header,
        len(header) * ["---"],
    ]
    rows = header + body
    return "\n".join(["|{}|".format("|".join(row)) for row in rows])


@experiment.with_experiment_storages()
def create_json(merge_exp, eval_exp, *fisher_exps):
    og_evals = eval_exp.retrieve_items_by_class(
        eval_execs.CheckpointEvaluationResults, run_uuid=None
    )
    print("Retrieved evaluation results.")
    uuid_to_fisher_item = {}
    for fisher_exp in fisher_exps:
        uuid_to_fisher_item.update(fisher_exp.retrieve_all_items())
        print("Retrieved Fisher items.")

    fisher_summaries = [
        item
        for item in uuid_to_fisher_item.values()
        if isinstance(item, fisher_execs.FisherMatricesSummary)
    ]

    uuid_to_saved_fisher_matrix = {
        k: v
        for k, v in uuid_to_fisher_item.items()
        if isinstance(v, fisher_execs.SavedFisherMatrix)
    }

    def fisher_blob_uuid_to_item_uuid(blob_uuid):
        for item_uuid, item in uuid_to_saved_fisher_matrix.items():
            if item.blob_uuid == blob_uuid:
                return item_uuid
        raise ValueError(f"Could not find SavedFisherMatrix with blob_uuid {blob_uuid}")

    def fisher_run_params(run_id):
        for fisher_exp in fisher_exps:
            try:
                return fisher_exp.retrieve_run_params(run_id)
            except ValueError:
                pass
        raise ValueError(f"Fisher run params for run {run_id} not found.")

    items = []
    for i, run_id in enumerate(merge_exp.retrieve_run_uuids(RunState.FINISHED)):
        print(f"Run {i}")
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

        target_mtm = params.models_to_merge[0]
        donor_mtm = params.models_to_merge[1]

        for fisher_summary in fisher_summaries:
            target_sfm_id = fisher_blob_uuid_to_item_uuid(target_mtm.fisher_matrix_uuid)
            donor_sfm_id = fisher_blob_uuid_to_item_uuid(donor_mtm.fisher_matrix_uuid)

            if target_sfm_id in fisher_summary.saved_fisher_matrix_uuids:
                target_fisher_epoch = fisher_summary.saved_fisher_matrix_uuids.index(
                    target_sfm_id
                )
            if donor_sfm_id in fisher_summary.saved_fisher_matrix_uuids:
                donor_fisher_epoch = fisher_summary.saved_fisher_matrix_uuids.index(
                    donor_sfm_id
                )

        target_fisher_params = fisher_run_params(target_mtm.fisher_run_uuid)
        donor_fisher_params = fisher_run_params(donor_mtm.fisher_run_uuid)

        items.append(
            {
                "task": target_mtm.task,
                "donor_task": donor_mtm.task,
                "donor_fisher_epoch": donor_fisher_epoch,
                "hyperparams": {
                    "target": {
                        "fisher_epoch": target_fisher_epoch,
                        "num_examples": target_fisher_params.num_examples,
                        "beta": target_fisher_params.variational_fisher_beta,
                        "learning_rate": target_fisher_params.learning_rate,
                    },
                    "donor": {
                        "fisher_epoch": donor_fisher_epoch,
                        "num_examples": donor_fisher_params.num_examples,
                        "beta": donor_fisher_params.variational_fisher_beta,
                        "learning_rate": donor_fisher_params.learning_rate,
                    },
                },
                "original_score": eval_res.results,
                "merged_score": res.results,
                "weighting": res.weighting[0],
            }
        )

    return items


def json_to_md_table(filepath, vertical=True, mc_score=None):
    items = _load_json(filepath)

    fisher_epochs = set()
    num_examples = set()
    betas = set()
    learning_rates = set()
    for item in items:
        donor_hp = item["hyperparams"]["donor"]
        fisher_epochs.add(donor_hp["fisher_epoch"])
        num_examples.add(donor_hp["num_examples"])
        betas.add(donor_hp["beta"])
        learning_rates.add(donor_hp["learning_rate"])

    fisher_epochs = sorted(fisher_epochs)
    num_examples = sorted(num_examples)
    betas = sorted(betas)
    learning_rates = sorted(learning_rates)

    rows = [
        ["Examples", "Beta", "LR"] + [str(c) for c in fisher_epochs],
    ]

    for ds_size in num_examples:
        for beta in betas:
            for lr in learning_rates:
                row = [str(ds_size), str(beta), str(lr)]
                for epoch in fisher_epochs:
                    donor_hp = {
                        "fisher_epoch": epoch,
                        "num_examples": ds_size,
                        "beta": beta,
                        "learning_rate": lr,
                    }
                    for item in items:
                        if item["hyperparams"]["donor"] == donor_hp:
                            # score = _get_single_score(item["merged_score"])
                            #
                            # score = _get_single_score(
                            #     item["merged_score"]
                            # ) - _get_single_score(item["original_score"])
                            #
                            score = _get_single_score(item["merged_score"]) - mc_score

                            score = str(round(score, 1))
                            row.append(score)
                rows.append(row)

    if vertical:
        rows = _tranpose(rows)

    return _to_md_table(rows)


if True:
    filepath = BERT_BASE_VAR_DIAG_RTE_MNLI_0_0003_PHASE_II_JSON
    t = json_to_md_table(filepath, mc_score=70.8)
    print(t)

    ###########################################################################

    # merge_exp = bert_base_var_diag_merge_phase_ii.RteMnli_BestCkpt_Iso_0003_MergeSame_PhaseII
    # fisher_exps = [
    #     bert_base_fisher_var_diag.RteMnliBestCkpt_Iso_0003_PhaseII,
    # ]
    # eval_exp = finetune_bert_base.GlueEval_Regs
    # summary = create_json(merge_exp, eval_exp, *fisher_exps)
    # s = json.dumps(summary, indent=2)
    # print(s)

    ###########################################################################

    # filepath = BERT_BASE_VAR_DIAG_RTE_MNLI_0_0003_PHASE_I_JSON
    # t = json_to_md_table(filepath, mc_score=70.8)
    # print(t)

    ###########################################################################

    # merge_exp = bert_base_var_diag_merge_phase_i.RteMnli_BestCkpt_Iso_0003_MergeSame_PhaseI
    # fisher_exps = [
    #     bert_base_fisher_var_diag.RteBestCkpt_Iso_0003_PhaseI,
    #     bert_base_fisher_var_diag.MnliBestCkpt_Iso_0003_PhaseI,
    # ]
    # eval_exp = finetune_bert_base.GlueEval_Regs
    # summary = create_json(merge_exp, eval_exp, *fisher_exps)
    # s = json.dumps(summary, indent=2)
    # print(s)
