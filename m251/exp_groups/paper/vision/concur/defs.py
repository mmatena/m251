"""TODO: Add title."""

# NOTE: Skipping caltech101 as I am having trouble downloading it.
TASKS = (
    "cifar100",
    "dtd",
    "oxford_iiit_pet",
    "cars196",
    "cifar10",
    "food101",
    "oxford_flowers102",
)

TASK_TO_NUM_TRIALS = {
    "cifar100": 5,
    "dtd": 5,
    "oxford_iiit_pet": 5,
}


# NOTE: Not using sun397 as it is big (and has a lot of classes).
TASK_TO_TRAIN_EXAMPLES = {
    "cars196": 8_144,
    "cifar10": 50_000,
    "cifar100": 50_000,
    "dtd": 1_880,
    "food101": 75_750,
    "oxford_iiit_pet": 3_680,
    "oxford_flowers102": 1_020,
    # "sun397": 76_128,
}


TASK_TO_FINETUNED_MODEL = {
    "cars196": " stanford_cars_1x",
    "cifar10": "cifar10_1x",
    "cifar100": "cifar100_1x",
    "dtd": "dtd_split1_1x",
    "food101": "food101_1x",
    "oxford_iiit_pet": "oxford_pets_1x",
    "oxford_flowers102": "oxford_102flowers_1x",
    # "sun397": "sun397_1x",
}


TASK_TO_4X_FINETUNED_MODEL = {
    "cars196": " stanford_cars_4x",
    "cifar10": "cifar10_4x",
    "cifar100": "cifar100_4x",
    "dtd": "dtd_split1_4x",
    "food101": "food101_4x",
    "oxford_iiit_pet": "oxford_pets_4x",
    "oxford_flowers102": "oxford_102flowers_4x",
    # "sun397": "sun397_4x",
}
