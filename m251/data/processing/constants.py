"""TODO: Add title."""

# Despite what the paper says, STS-B starts at 0, not 1.
STSB_MIN = 0
STSB_MAX = 5
# Corresponds to rounding to nearest 0.2 increment.
STSB_NUM_BINS = 5 * (STSB_MAX - STSB_MIN)

# NOTE: There are 8 unique tasks total (not counting WNLI).
NUM_GLUE_LABELS = {
    "cola": 2,
    "mnli": 3,
    "mnli_matched": 3,
    "mnli_mismatched": 3,
    "mrpc": 2,
    "sst2": 2,
    "stsb": STSB_NUM_BINS,
    "qqp": 2,
    "qnli": 2,
    "rte": 2,
    "wnli": 2,
}

NUM_GLUE_TRAIN_EXAMPLES = {
    "cola": 8_551,
    "mnli": 392_702,
    "mrpc": 3_668,
    "sst2": 67_349,
    "stsb": 5_749,
    "qqp": 363_849,
    "qnli": 104_743,
    "rte": 2_490,
    "wnli": 635,
}


# Taken from page 19 of https://arxiv.org/pdf/2005.00770.pdf. The
# tasks in the values are ordered in descending order of benefit.
#
# NOTE: Only includes transfer from GLUE tasks. Only tasks with a
# "significant enough" benefit are included.
GLUE_POSITIVE_TRANSFER_TASKS = {
    "rte": ("mnli", "qnli", "stsb"),
    "mrpc": ("qqp", "mnli", "stsb", "qnli", "cola"),
    "stsb": ("qnli", "qqp"),
    "cola": ("sst2",),
    #
    "sst2": ("mnli", "qnli"),
    "qqp": ("cola",),
    "mnli": ("qnli", "rte", "mrpc"),
    "qnli": ("mnli",),
}
