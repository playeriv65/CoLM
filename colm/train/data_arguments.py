from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class DataArguments:
    train_files: list[str] = field(
        default_factory=lambda: ["data/MathInstruct.jsonl"],
        metadata={"help": "Training data files (jsonl) or a hub dataset id."},
    )
    data_dir: str = field(default="data", metadata={"help": "Directory of the local data files."})
    hf_datasets_cache_dir: str | None = field(
        default=None, metadata={"help": "datasets cache override (default: $HF_HOME)."}
    )
    subset_selection: Literal[
        "random",
        "balanced_longest_selection",
        "longest_sourcewise_selection",
        "longest_selection",
        "use_small_sources",
    ] = field(
        default="use_small_sources",
        metadata={"help": "How the `percentage` of the data is chosen (unused at percentage 1)."},
    )
    sample_data_seed: int = field(default=42, metadata={"help": "Seed used for data sampling."})
    percentage: float = field(default=1.0, metadata={"help": "Sampling percentage of the data."})
    subset_index_files: list[str] = field(
        default_factory=list, metadata={"help": "Files with subset indices to train on."}
    )
    output_root: str = field(
        default="out", metadata={"help": "Parent directory of auto-named output directories."}
    )


def get_data_statistics(lm_datasets, is_custom_dataset=False):
    """Print the number of examples and the average (completion) length."""

    def get_length(examples):
        lengths = [len(ids) for ids in examples["input_ids"]]
        completion_lens = [(torch.tensor(labels) > -1).sum() for labels in examples["labels"]]
        return {"length": lengths, "c_length": completion_lens}

    if not isinstance(lm_datasets, dict):
        lm_datasets = {"train": lm_datasets}

    for key, dataset in lm_datasets.items():
        data_size = len(dataset)
        if not is_custom_dataset:
            dataset = dataset.map(get_length, batched=True)
            lengths = dataset["length"]
            c_lengths = dataset["c_length"]
        else:
            lengths = [len(example["input_ids"]) for example in dataset]
            c_lengths = [len(example["labels"]) for example in dataset]
        print(f"[{key} set] examples: {data_size}; # avg tokens: {sum(lengths) / len(lengths)}")
        print(
            f"[{key} set] examples: {data_size}; # avg completion tokens: {sum(c_lengths) / len(c_lengths)}"
        )
