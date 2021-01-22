"""TODO: Add title."""

HIGH_RESOURCE_TASKS = {"mnli", "sst2", "qqp", "qnli"}

LOW_RESOURCE_TASKS = {"cola", "mrpc", "stsb", "rte"}
LOW_RESOURCE_TRIALS = 5

TASKS_FINETUNED_FROM_MNLI = {"mrpc", "stsb", "rte"}


TASK_TO_CKPT_BERT_BASE = {
    "cola": "textattack/bert-base-uncased-CoLA",
    "mnli": "textattack/bert-base-uncased-MNLI",
    "mrpc": "textattack/bert-base-uncased-MRPC",
    "sst2": "textattack/bert-base-uncased-SST-2",
    "stsb": "textattack/bert-base-uncased-STS-B",
    "qqp": "textattack/bert-base-uncased-QQP",
    "qnli": "textattack/bert-base-uncased-QNLI",
    "rte": "textattack/bert-base-uncased-RTE",
    #
    "squad2": "twmkn9/bert-base-uncased-squad2",
}

BERT_BASE_MNLI_CKPT = TASK_TO_CKPT_BERT_BASE["mnli"]

LABEL_MAP_OVERRIDES = {
    "mnli": (1, 2, 0),
    "mnli_matched": (1, 2, 0),
    "mnli_mismatched": (1, 2, 0),
}


# For SQuAD2:
#   twmkn9/bert-base-uncased-squad2


TASK_TO_CKPT_ROBERTA_LARGE = {
    # "cola": "textattack/bert-base-uncased-CoLA",
    # "mnli": "textattack/bert-base-uncased-MNLI",
    # "mrpc": "textattack/bert-base-uncased-MRPC",
    # "sst2": "textattack/bert-base-uncased-SST-2",
    # "stsb": "textattack/bert-base-uncased-STS-B",
    # "qqp": "textattack/bert-base-uncased-QQP",
    # "qnli": "textattack/bert-base-uncased-QNLI",
    # "rte": "textattack/bert-base-uncased-RTE",
    # #
    # "squad2": "twmkn9/bert-base-uncased-squad2",
}
