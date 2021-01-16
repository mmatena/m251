"""S2ORC: The Semantic Scholar Open Research Corpus

See https://github.com/allenai/s2orc for details.

See https://github.com/allenai/dont-stop-pretraining/issues/4 for details
on how the Don't Stop Pretraining paper processed this data set.
"""
import json
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from del8.core.di import executable
from del8.core.di import scopes

from del8.executables.data import preprocessing as preprocessing_execs
from del8.executables.data import tfds as tfds_execs

from m251.data.processing import mlm
from m251.models.bert import bert as bert_common


_CITATION = R"""\
@inproceedings{lo-wang-2020-s2orc,
    title = "{S}2{ORC}: The Semantic Scholar Open Research Corpus",
    author = "Lo, Kyle  and Wang, Lucy Lu  and Neumann, Mark  and Kinney, Rodney  and Weld, Daniel",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.447",
    doi = "10.18653/v1/2020.acl-main.447",
    pages = "4969--4983"
}"""

_DESCRIPTION = "See the paper and github."

_HOMEPAGE_URL = "https://github.com/allenai/s2orc"


_DOWNLOAD_LINKS_FILE = os.path.join(
    os.path.dirname(__file__), "s2orc_download_links.json"
)


_DOMAIN_TO_FOS = {
    "bio_med": ["Biology", "Medicine"],
    "cs": ["Computer Science"],
}


def _get_all_links():
    with open(_DOWNLOAD_LINKS_FILE, "r") as f:
        return json.load(f)


class S2orcConfig(tfds.core.BuilderConfig):
    def __init__(self, domain, s2orc_version, num_shards, **kwargs):
        name = f"{s2orc_version}.{domain}.{num_shards}"
        super().__init__(
            name=name,
            description=_DESCRIPTION,
            version=tfds.core.Version("0.0.1"),
            release_notes={
                "0.0.1": "First version.",
            },
            **kwargs,
        )
        self.domain = domain
        self.s2orc_version = s2orc_version
        # Each full shard is approx 529MB + 1.6GB zipped & 2.0GB + 6.5GB unzipped
        self.num_shards = num_shards

    def get_metadata_links(self):
        links = _get_all_links()
        return [links[f"metadata_{i}"] for i in range(self.num_shards)]

    def get_pdf_parses_links(self):
        links = _get_all_links()
        return [links[f"pdf_parses_{i}"] for i in range(self.num_shards)]


class S2orc(tfds.core.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        S2orcConfig(
            s2orc_version="20200705v1",
            domain="bio_med",
            num_shards=5,
        ),
        S2orcConfig(
            s2orc_version="20200705v1",
            domain="cs",
            num_shards=5,
        ),
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "abstract_paragraphs": tfds.features.Sequence(tfds.features.Text()),
                    "body_text_paragraphs": tfds.features.Sequence(
                        tfds.features.Text()
                    ),
                }
            ),
            homepage=_HOMEPAGE_URL,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        file_paths = dl_manager.download_and_extract(
            {
                "metadata": self.builder_config.get_metadata_links(),
                "pdf_parses": self.builder_config.get_pdf_parses_links(),
            }
        )

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs=file_paths,
            ),
        ]

    def _get_paper_ids(self, metadata_file):
        paper_ids = set()
        domain_foses = _DOMAIN_TO_FOS[self.builder_config.domain]
        with open(metadata_file, "r") as f:
            for line in f:
                metadata_dict = json.loads(line)
                foses = metadata_dict["mag_field_of_study"] or []
                has_body = metadata_dict.get("has_pdf_parsed_body_text", False)
                has_abstract = metadata_dict.get("has_pdf_parsed_abstract", False)

                if (has_body or has_abstract) and any(
                    fos in domain_foses for fos in foses
                ):
                    paper_ids.add(metadata_dict["paper_id"])
        return paper_ids

    def _generate_examples(self, metadata, pdf_parses):
        for metadatum, pdf_parse in zip(metadata, pdf_parses):
            paper_ids = self._get_paper_ids(metadatum)
            with open(pdf_parse, "r") as f:
                for line in f:
                    item = json.loads(line)
                    paper_id = item["paper_id"]
                    if paper_id not in paper_ids:
                        continue
                    elif not item["abstract"] and not item["body_text"]:
                        continue
                    yield paper_id, {
                        "abstract_paragraphs": [p["text"] for p in item["abstract"]],
                        "body_text_paragraphs": [p["text"] for p in item["body_text"]],
                    }


###############################################################################


def _to_paragraphs(ds, max_paragraphs=32):
    def map_fn(x):
        print(x["abstract_paragraphs"])
        paragraphs = tf.concat(
            [x["abstract_paragraphs"], x["body_text_paragraphs"]], axis=0
        )
        paragraphs = tf.random.shuffle(paragraphs)
        paragraphs = paragraphs[:max_paragraphs]
        return {"text": paragraphs}

    def non_empty(x):
        return tf.strings.length(x["text"]) > 0

    ds = ds.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
    ds = ds.filter(non_empty)
    return ds


###############################################################################


@executable.executable()
def to_paragraphs(dataset, max_paragraphs_per_doc=32):
    return _to_paragraphs(dataset, max_paragraphs=max_paragraphs_per_doc)


###############################################################################


@executable.executable()
def dataset_name(task):
    return f"s2orc/20200705v1.{task}.5"


@executable.executable(
    default_bindings={
        "dataset_name": dataset_name,
        "tfds_dataset": tfds_execs.tfds_dataset,
        "preprocessor": to_paragraphs,
        "mlm_preprocessor": mlm.mlm_preprocessor,
        "common_prebatch_processer": preprocessing_execs.common_prebatch_processer,
        "batcher": preprocessing_execs.batcher,
        "tokenizer": bert_common.bert_tokenizer,
    }
)
def mlm_dataset(
    _tfds_dataset,
    _preprocessor,
    _mlm_preprocessor,
    _common_prebatch_processer,
    _batcher,
):
    ds = _tfds_dataset()
    ds = _preprocessor(ds)
    ds = _mlm_preprocessor(ds)
    ds = _common_prebatch_processer(ds)
    ds = _batcher(ds)
    return ds
