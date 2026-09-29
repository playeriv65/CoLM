# Adopted from https://github.com/princeton-nlp/MeZO/blob/main/large_models/tasks.py
import logging
import sys
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from colm.data.templates import (
    BoolQTemplate,
    BoolQTemplateV2,
    BoolQTemplateV3,
    CBTemplate,
    CopaTemplate,
    DROPTemplate,
    MultiRCTemplate,
    ReCoRDTemplateGPT3,
    RTETemplate,
    SQuADv2Template,
    SST2Template,
    Template,
    WICTemplate,
    WSCTemplate,
)
from colm.data.utils import temp_seed

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# datasets >= 4 only resolves namespaced hub ids (the bare names were loading scripts).
SUPERGLUE = "aps/super_glue"
GLUE = "nyu-mll/glue"
BOOLQ = "google/boolq"
SQUAD = "rajpurkar/squad"
DROP = "ucinlp/drop"


def get_task(task_name):
    aa = task_name.split("__")
    if len(aa) == 2:
        task_group, subtask = aa
    else:
        task_group = aa[0]
        subtask = None
    class_ = getattr(sys.modules[__name__], f"{task_group}Dataset")
    instance = class_(subtask)
    return instance


@dataclass
class Sample:
    id: int = None
    data: dict = None
    correct_candidate: str | list[str] = None
    candidates: list[str] = None


class Dataset:
    train_sep = "\n\n"
    generation = False  # whether this is a generation task
    classification = True  # whether train as classification

    def __init__(self, subtask=None, **kwargs) -> None:
        self.subtask = subtask

    def load_dataset():
        raise NotImplementedError

    def get_template(self, template_version=0):
        templates = {0: Template}
        return templates[template_version]

    def build_sample(self, example):
        return

    def sample_subset(self, data_split="train", seed=0, num=100):
        """`num` samples of the split in a seeded random order; all of them for `num` <= 0."""
        with temp_seed(seed):
            samples = self.samples[data_split]
            index = np.random.permutation(len(samples)).tolist()
            if num > 0:
                index = index[:num]
            return [samples[i] for i in index]

    @property
    def valid_samples(self):
        return self.samples["valid"]


class SST2Dataset(Dataset):
    train_sep = "\n\n"

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset(GLUE, "sst2")
        train_d = d["train"]
        validation_d = d["validation"]

        train_samples = [self.build_sample(example) for example in train_d]
        valid_samples = [self.build_sample(example) for example in validation_d]

        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        label = int(example["label"])
        return Sample(id=example["idx"], data=example, correct_candidate=label, candidates=[0, 1])

    def get_template(self, template_version=0):
        return {0: SST2Template}[template_version]()


class CopaDataset(Dataset):
    train_sep = "\n\n"
    classification = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        train_examples = load_dataset(SUPERGLUE, "copa")["train"]
        valid_examples = load_dataset(SUPERGLUE, "copa")["validation"]

        train_samples = [self.build_sample(example) for example in train_examples]
        valid_samples = [self.build_sample(example) for example in valid_examples]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example):
        sample = Sample(
            id=example["idx"],
            data=example,
            candidates=[example["choice1"], example["choice2"]],
            correct_candidate=example[f"choice{example['label'] + 1}"],
        )

        return sample

    def get_template(self, template_version=0):
        return {0: CopaTemplate}[template_version]()


class BoolQDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset(BOOLQ)
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example,
            candidates=["Yes", "No"],
            correct_candidate="Yes" if example["answer"] else "No",
        )

        return sample

    def get_template(self, template_version=2):
        return {0: BoolQTemplate, 1: BoolQTemplateV2, 2: BoolQTemplateV3}[template_version]()


class MultiRCDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset(SUPERGLUE, "multirc")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(data=example, candidates=[0, 1], correct_candidate=example["label"])

        return sample

    def get_template(self, template_version=0):
        return {0: MultiRCTemplate}[template_version]()


class CBDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset(SUPERGLUE, "cb")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(data=example, candidates=[0, 1, 2], correct_candidate=example["label"])

        return sample

    def get_template(self, template_version=0):
        return {0: CBTemplate}[template_version]()


class WICDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset(SUPERGLUE, "wic")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(data=example, candidates=[0, 1], correct_candidate=example["label"])

        return sample

    def get_template(self, template_version=0):
        return {0: WICTemplate}[template_version]()


class WSCDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset(SUPERGLUE, "wsc.fixed")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(data=example, candidates=[0, 1], correct_candidate=example["label"])

        return sample

    def get_template(self, template_version=0):
        return {0: WSCTemplate}[template_version]()


class ReCoRDDataset(Dataset):
    classification = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset(SUPERGLUE, "record")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(
            data=example, candidates=example["entities"], correct_candidate=example["answers"]
        )

        return sample

    def get_template(self, template_version=0):
        return {0: ReCoRDTemplateGPT3}[template_version]()


class RTEDataset(Dataset):
    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset(subtask, **kwargs)

    def load_dataset(self, path, **kwargs):
        d = load_dataset(SUPERGLUE, "rte")
        train_set = d["train"]
        valid_set = d["validation"]

        train_samples = [self.build_sample(example) for example in train_set]
        valid_samples = [self.build_sample(example) for example in valid_set]
        self.samples = {"train": train_samples, "valid": valid_samples}

    def build_sample(self, example):
        sample = Sample(data=example, candidates=[0, 1], correct_candidate=example["label"])

        return sample

    def get_template(self, template_version=0):
        return {0: RTETemplate}[template_version]()


class SQuADDataset(Dataset):
    metric_name = "f1"
    generation = True
    classification = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset(SQUAD)
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [
            self.build_sample(example, idx) for idx, example in enumerate(train_examples)
        ]
        valid_samples = [
            self.build_sample(example, idx) for idx, example in enumerate(valid_examples)
        ]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example["answers"]["text"]
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "title": example["title"],
                "context": example["context"],
                "question": example["question"],
                "answers": answers,
            },
            candidates=None,
            correct_candidate=answers,
        )

    def get_template(self, template_version=0):
        return {0: SQuADv2Template}[template_version]()


class DROPDataset(Dataset):
    metric_name = "f1"
    generation = True
    classification = False

    def __init__(self, subtask=None, **kwargs) -> None:
        self.load_dataset()

    def load_dataset(self):
        dataset = load_dataset(DROP)
        train_examples = dataset["train"]
        valid_examples = dataset["validation"]

        train_samples = [
            self.build_sample(example, idx) for idx, example in enumerate(train_examples)
        ]
        valid_samples = [
            self.build_sample(example, idx) for idx, example in enumerate(valid_examples)
        ]
        self.samples = {"train": train_samples, "valid": valid_samples}

    # for generative tasks, candidates are []
    def build_sample(self, example, idx):
        answers = example["answers_spans"]["spans"]
        assert len(answers) > 0
        return Sample(
            id=idx,
            data={
                "context": example["passage"],
                "question": example["question"],
                "answers": answers,
            },
            candidates=None,
            correct_candidate=answers,
        )

    def get_template(self, template_version=0):
        return {0: DROPTemplate}[template_version]()
