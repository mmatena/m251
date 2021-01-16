"""TODO: Add title."""
import csv
import io
import json
import os


_RESULTS_DIR = "~/Desktop/projects/m251_1st_paper_data"
RESULTS_DIR = os.path.expanduser(_RESULTS_DIR)


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
