"""TODO: Add title."""

# Despite what the paper says, STS-B starts at 0, not 1.
STSB_MIN = 0
STSB_MAX = 5
# Corresponds to rounding to nearest 0.2 increment.
STSB_NUM_BINS = 5 * (STSB_MAX - STSB_MIN)

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
