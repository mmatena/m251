"""TODO: Add title."""

HIGH_RESOURCE_TASKS = {"mnli", "sst2", "qqp", "qnli"}
HIGH_RESOURCE_TRIALS = 1

LOW_RESOURCE_TASKS = {"cola", "mrpc", "stsb", "rte"}
LOW_RESOURCE_TRIALS = 5


BAD_FINETUNE_RUN_UUIDS = frozenset(
    {
        "37dbf11090b047b2ba2e9996597e22ab",
        "ab6ce15a17ad4ea287c08093270ee494",
        "b8103c8e19054604a420b1ec2c1e4a15",
        "2b23839254934890acd9fab09803382c",
    }
)
