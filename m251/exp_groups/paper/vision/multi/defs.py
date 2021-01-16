"""TODO: Add title."""

# NOTE: Skipping caltech101 as I am having trouble downloading it.
TASKS = ("cifar100", "dtd", "oxford_iiit_pet")

TASK_TO_NUM_TRIALS = {
    "cifar100": 5,
    "dtd": 5,
    "oxford_iiit_pet": 5,
}


TASK_TO_TRAIN_EXAMPLES = {
    "cifar100": 50_000,
    "dtd": 1_880,
    "oxford_iiit_pet": 3_680,
}
