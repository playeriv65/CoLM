"""SuperGLUE-style task data (MeZO tasks): few-shot samples rendered by the task templates.

Two training modes: generation-style tasks (the loss covers the last `option_len` tokens of each
example) work with every trainer; classification-style tasks (each example is a list of candidate
sequences, the loss is a cross-entropy over the candidates' option log-likelihoods) work with the
full-batch baseline only.
"""

import json
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

import colm.data.utils as utils
from colm.data.get_training_dataset import ExampleCollator, PackCollator
from colm.data.tasks import Sample, get_task
from colm.selection.packing import IGNORE_INDEX, Example

logger = logging.getLogger(__name__)


class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Classification examples are lists of candidate sequences.
        sequences = [c for x in data for c in (x if isinstance(x, list) else [x])]
        self.mean_tokens = sum(len(c["input_ids"]) for c in sequences) / len(sequences)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def convert_samples(samples, task, tokenizer, context_length, only_train_option):
    """Tokenised training examples of task samples.

    A sample with a sequence above `context_length` tokens is dropped (counted and logged), never
    truncated.
    """
    data, dropped = [], 0
    for index, sample in enumerate(tqdm(samples, mininterval=10)):
        try:
            encoded, option_lens = utils.encode_prompt(
                task,
                task.get_template(),
                [],
                sample,
                tokenizer,
                context_length,
                generation=task.generation,
                generation_with_gold=True,
            )
        except utils.PromptTooLong:
            dropped += 1
            continue
        if task.generation:
            correct = 0
        elif isinstance(sample.correct_candidate, list):
            correct = sample.candidates.index(sample.correct_candidate[0])
        else:
            correct = sample.candidates.index(sample.correct_candidate)
        source = sample.data.get("source", 0) if sample.data else 0
        if task.classification:
            # The label is the correct candidate; every candidate is one sequence.
            data.append(
                [
                    {
                        "input_ids": encoded[i],
                        "labels": correct,
                        "option_len": option_lens[i],
                        "num_options": len(sample.candidates),
                    }
                    for i in range(len(encoded))
                ]
            )
        else:
            item = {"input_ids": encoded[correct], "sources": source, "indices": index}
            if only_train_option:
                item["option_len"] = option_lens[correct]
            data.append(item)
    logger.info(
        f"Dropped {dropped} of {len(samples)} samples longer than {context_length} tokens "
        "(never truncated)"
    )
    return data


def option_examples(features: list[dict]) -> list[Example]:
    """Packed-batch examples of generation-style samples: only the option is a label."""
    examples = []
    for f in features:
        ids = np.array(f["input_ids"])
        labels = ids.copy()
        if "option_len" in f:
            labels[: len(ids) - f["option_len"]] = IGNORE_INDEX
        examples.append(
            Example(ids, labels, int(f["sources"]), f["indices"], f.get("option_len", -1))
        )
    return examples


@dataclass
class ClassificationCollator:
    """Candidates of several examples as one padded batch (`num_options` says how they group)."""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        flat = [c for candidates in features for c in candidates]
        ids = [torch.tensor(c["input_ids"]) for c in flat]
        lengths = torch.tensor([len(x) for x in ids])
        input_ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        return {
            "input_ids": input_ids,
            "attention_mask": (torch.arange(input_ids.shape[1]) < lengths[:, None]).long(),
            "labels": torch.tensor([c["labels"] for c in flat]),
            "option_len": torch.tensor([c["option_len"] for c in flat]),
            "num_options": torch.tensor([c["num_options"] for c in flat]),
        }


def classification_loss(logits, batch) -> torch.Tensor:
    """Cross-entropy over the candidates of each example, scored by their mean option log-probability.

    The option is the last `option_len` real tokens of a candidate (the upstream code counted them
    back from the padded width, so the option of a shorter candidate lost its first tokens).
    """
    input_ids, mask = batch["input_ids"], batch["attention_mask"]
    targets = input_ids[:, 1:]
    positions = torch.arange(targets.shape[1], device=targets.device)
    real = mask.sum(dim=1, keepdim=True)
    option_len = batch["option_len"].unsqueeze(1)
    keep = (positions >= real - 1 - option_len) & (positions < real - 1)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    picked = torch.gather(log_probs, -1, targets.unsqueeze(-1)).squeeze(-1)
    score = (picked * keep).sum(-1) / keep.sum(-1)  # one number per candidate

    num_options, labels = batch["num_options"].tolist(), batch["labels"]
    losses, start = [], 0
    while start < len(num_options):
        end_ = start + num_options[start]
        losses.append(F.cross_entropy(score[start:end_].unsqueeze(0), labels[start : start + 1]))
        start = end_
    return torch.stack(losses).mean()


def build_superglue(data_args, training_args, tokenizer, context_length):
    """(train dataset, collator, None) of a `superglue-<task>` / `load-superglue-<task>` run."""
    name = data_args.train_files[0]
    task = get_task(name.split("-")[-1])
    if name.split("-")[0] == "load":
        with open(os.path.join(data_args.data_dir, f"{name}.jsonl")) as f:
            samples = [Sample(**json.loads(line)) for line in f]
    else:
        samples = task.sample_subset(num=1000)
    data = convert_samples(
        samples,
        task,
        tokenizer,
        context_length,
        training_args.only_train_option,
    )
    logger.info(
        f"{len(data)} of {len(samples)} {name} examples kept (generation={task.generation}, "
        f"classification={task.classification})"
    )
    if task.classification:
        if training_args.coreset:
            raise ValueError("classification tasks train with data_selection_method=none only")
        collator = ClassificationCollator(tokenizer.pad_token_id)
    else:
        collator = (ExampleCollator if training_args.coreset else PackCollator)(option_examples)
    return ListDataset(data), collator, None
