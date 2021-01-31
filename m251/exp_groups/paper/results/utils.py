"""TODO: Add title."""
import csv
import io
import json
import os


_RESULTS_DIR = "~/Desktop/projects/m251_1st_paper_data"
RESULTS_DIR = os.path.expanduser(_RESULTS_DIR)

GLUE_TASKS_ORDER = ("cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte")

TASK_NICE_NAMES = {
    "cola": "CoLA",
    "sst2": "SST-2",
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "qqp": "QQP",
    "mnli": "MNLI",
    "qnli": "QNLI",
    "rte": "RTE",
    "squad2": "SQuAD",
}


def result_file(subpath):
    return os.path.join(RESULTS_DIR, subpath)


def load_json(json_file):
    if not isinstance(json_file, str):
        # Assume this is the actual json object.
        return json_file
    json_file = os.path.expanduser(json_file)
    with open(json_file, "r") as f:
        return json.load(f)


def get_single_score(scores):
    if not isinstance(scores, dict):
        return scores
    values = [get_single_score(v) for v in scores.values()]
    return sum(values) / len(values)


def csv_to_str(nested_lists):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(nested_lists)
    return output.getvalue()


def table_to_latex(nested_lists):
    # Items of nested_lists that are string are row literals.
    rows = [r if isinstance(r, str) else " & ".join(r) + R" \\" for r in nested_lists]
    return "\n".join(rows)
