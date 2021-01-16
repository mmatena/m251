"""TFDS for the target tasks."""
import json

import tensorflow as tf
import tensorflow_datasets as tfds

from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.data import preprocessing as preprocessing_execs
from del8.executables.data import tfds as tfds_execs

from m251.data.processing import mixture
from m251.data.processing import mlm
from m251.models.bert import bert as bert_common


NUM_CLASSES = {
    "chemprot": 13,
    "acl_arc": 6,
    "sci_erc": 7,
    "hyperpartisan": 2,
    "helpfulness": 2,
}

TRAIN_EXAMPLES = {
    "chemprot": 4_169,
    "acl_arc": 1_688,
    "sci_erc": 3_219,
    "hyperpartisan": 516,
    "helpfulness": 115_251,
}

TASK_TO_DAPT_NAME = {
    "chemprot": "allenai/biomed_roberta_base",
    "acl_arc": "allenai/cs_roberta_base",
    "sci_erc": "allenai/cs_roberta_base",
    "hyperpartisan": "allenai/news_roberta_base",
    "helpfulness": "allenai/reviews_roberta_base",
}


# Using the RoBERTa-base tokenizer (or whatever came with the paper's huggingface models.)
MAX_SEQUENCE_LENGTHS = {
    "chemprot": 375,
    "acl_arc": 280,
    "sci_erc": 135,
    "hyperpartisan": 5354,
    "helpfulness": 7196,
}


_TASK_TO_S3_NAME = {
    "chemprot": "chemprot",
    "acl_arc": "citation_intent",
    "sci_erc": "sciie",
    "hyperpartisan": "hyperpartisan_news",
    "helpfulness": "amazon",
}

TASK_CLASS_LABELS = {
    "chemprot": [
        "ACTIVATOR",
        "AGONIST",
        "AGONIST-ACTIVATOR",
        "AGONIST-INHIBITOR",
        "ANTAGONIST",
        "DOWNREGULATOR",
        "INDIRECT-DOWNREGULATOR",
        "INDIRECT-UPREGULATOR",
        "INHIBITOR",
        "PRODUCT-OF",
        "SUBSTRATE",
        "SUBSTRATE_PRODUCT-OF",
        "UPREGULATOR",
    ],
    "acl_arc": [
        "Background",
        "CompareOrContrast",
        "Extends",
        "Future",
        "Motivation",
        "Uses",
    ],
    "sci_erc": [
        "COMPARE",
        "CONJUNCTION",
        "EVALUATE-FOR",
        "FEATURE-OF",
        "HYPONYM-OF",
        "PART-OF",
        "USED-FOR",
    ],
    "hyperpartisan": ["false", "true"],
    "helpfulness": ["unhelpful", "helpful"],
}

_CITATION = R"""\
@article{gururangan2020don,
  title={Don't Stop Pretraining: Adapt Language Models to Domains and Tasks},
  author={Gururangan, Suchin and Marasovi{\'c}, Ana and Swayamdipta, Swabha and Lo, Kyle and Beltagy, Iz and Downey, Doug and Smith, Noah A},
  journal={arXiv preprint arXiv:2004.10964},
  year={2020}
}"""

_DESCRIPTION = "See the paper and github."

_HOMEPAGE_URL = "https://github.com/allenai/dont-stop-pretraining"

_DOWNLOAD_URL = "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/{task}/{split}.jsonl"


class DomainTargetTaskConfig(tfds.core.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(
            version=tfds.core.Version("0.0.1"),
            release_notes={
                "0.0.1": "First version.",
            },
            **kwargs,
        )
        self.label_classes = TASK_CLASS_LABELS[self.name]
        self.urls = {
            split: _DOWNLOAD_URL.format(task=_TASK_TO_S3_NAME[self.name], split=split)
            for split in ["train", "dev", "test"]
        }


class DomainTargetTask(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        DomainTargetTaskConfig(name=name, description=_DESCRIPTION)
        for name in _TASK_TO_S3_NAME.keys()
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "text": tfds.features.Text(),
                    "label": tfds.features.ClassLabel(
                        names=self.builder_config.label_classes,
                    ),
                }
            ),
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        file_paths = dl_manager.download(
            {
                "train": self.builder_config.urls["train"],
                "validation": self.builder_config.urls["dev"],
                "test": self.builder_config.urls["test"],
            }
        )

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "file_path": file_paths["train"],
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "file_path": file_paths["validation"],
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "file_path": file_paths["test"],
                },
            ),
        ]

    def _generate_examples(self, file_path):
        with open(file_path, "r") as f:
            for index, line in enumerate(f):
                row = json.loads(line)
                example = {
                    "text": row["text"],
                    "label": row["label"],
                }
                row_id = row["id"] if "id" in row else index
                yield row_id, example


###############################################################################


def _convert_dataset_to_features(dataset, tokenizer, max_length):
    pad_token = tokenizer.pad_token_id
    # NOTE: Not sure if this is correct, but it matches up for BERT. RoBERTa does
    # not appear to use token types.
    pad_token_segment_id = tokenizer.pad_token_type_id

    def py_map_fn(text):
        text = tf.compat.as_str(text.numpy())

        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            truncation=True,
            return_tensors="tf",
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        return input_ids, token_type_ids

    def map_fn(example):
        input_ids, token_type_ids = tf.py_function(
            func=py_map_fn,
            inp=[example["text"]],
            Tout=[tf.int32, tf.int32],
        )
        return tf.squeeze(input_ids, 0), tf.squeeze(token_type_ids, 0), example["label"]

    def pad_fn(input_ids, token_type_ids, label):
        # Zero-pad up to the sequence length.
        padding_length = max_length - tf.shape(input_ids)[-1]
        ones_pad = tf.ones(padding_length, dtype=tf.int32)

        input_ids = tf.concat([input_ids, pad_token * ones_pad], axis=-1)
        token_type_ids = tf.concat(
            [token_type_ids, pad_token_segment_id * ones_pad], axis=-1
        )

        tf_example = {
            # Ensure the shape is known as this is often needed for downstream steps.
            "input_ids": tf.reshape(input_ids, [max_length]),
            "token_type_ids": tf.reshape(token_type_ids, [max_length]),
        }
        return tf_example, label

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(pad_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


###############################################################################


@executable.executable()
def preprocessor(dataset, task, tokenizer, sequence_length):
    return _convert_dataset_to_features(
        dataset,
        tokenizer,
        max_length=sequence_length,
    )


###############################################################################


@executable.executable()
def dataset_name(task):
    return f"domain_target_task/{task}"


@executable.executable(
    default_bindings={
        "tfds_dataset": tfds_execs.tfds_dataset,
        "preprocesser": preprocessor,
        "common_prebatch_processer": preprocessing_execs.common_prebatch_processer,
        "mixer": mixture.no_shuffle_mixer,
        "batcher": preprocessing_execs.batcher,
        "tokenizer": bert_common.bert_tokenizer,
        "dataset_name": dataset_name,
    }
)
def finetuning_dataset(
    tasks,
    _tfds_dataset,
    _preprocesser,
    _common_prebatch_processer,
    _mixer,
    _batcher,
    _dataset_name,
):
    datasets = []
    for task in tasks:
        task_bindings = [
            ("task", task),
            ("dataset_name", _dataset_name(task)),
        ]
        with scopes.binding_by_name_scopes(task_bindings):
            ds = _tfds_dataset()
            ds = _preprocesser(ds)
            ds = _common_prebatch_processer(ds)
            datasets.append(ds)

    mixture = _mixer(datasets)
    mixture = _batcher(mixture)

    return mixture


@executable.executable(
    default_bindings={
        "dataset_name": dataset_name,
        "tfds_dataset": tfds_execs.tfds_dataset,
        "mlm_preprocessor": mlm.mlm_preprocessor,
        "common_prebatch_processer": preprocessing_execs.common_prebatch_processer,
        "batcher": preprocessing_execs.batcher,
        "tokenizer": bert_common.bert_tokenizer,
    }
)
def mlm_dataset(
    _tfds_dataset,
    _mlm_preprocessor,
    _common_prebatch_processer,
    _batcher,
):
    ds = _tfds_dataset()
    ds = _mlm_preprocessor(ds)
    ds = _common_prebatch_processer(ds)
    ds = _batcher(ds)
    return ds


@executable.executable()
def glue_robust_evaluation_dataset_provider(_glue_robust_evaluation_dataset):
    return _glue_robust_evaluation_dataset


@executable.executable(
    default_bindings={
        "tfds_dataset": tfds_execs.tfds_dataset,
        "preprocesser": preprocessor,
        "tokenizer": bert_common.bert_tokenizer,
        "dataset_name": dataset_name,
        "glue_robust_evaluation_dataset_provider": glue_robust_evaluation_dataset_provider,
    }
)
def robust_evaluation_dataset(
    tasks,
    _glue_robust_evaluation_dataset_provider,
):
    from m251.data.glue import glue

    # NOTE: We are doing it this way rather than a default binding to prevent an
    # issue with circular imports.
    with scopes.binding_by_name_scope(
        "glue_robust_evaluation_dataset", glue.glue_robust_evaluation_dataset
    ):
        pass
        # NOTE: We are simply changing the default bindings and using the
        # version I wrote for GLUE. We should probably move that executable
        # to a common place.
        return _glue_robust_evaluation_dataset_provider()(tasks)
