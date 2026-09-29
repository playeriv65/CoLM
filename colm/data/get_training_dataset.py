import logging
import os
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset

import colm.data.utils as utils
from colm.selection.packing import Example, pack

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)


def get_training_dataset(
    train_files: list[str],
    tokenizer,
    context_length: int,
    sample_percentage=1.0,
    subset_index_files=None,
    template_variation=False,
    seed=0,
    hf_datasets_cache_dir=None,
    subset_selection="use_small_sources",
):
    """Training data of the files, tokenised without truncation.

    `context_length` is the context window of the model: an example above it is dropped (counted
    and logged per source), never cut.
    """
    raw_datasets = load_raw_dataset(
        train_files,
        sample_percentage=sample_percentage,
        subset_index_files=subset_index_files,
        seed=seed,
        cache_dir=hf_datasets_cache_dir,
        subset_selection=subset_selection,
    )

    if "instruction" in raw_datasets.column_names:
        lm_datasets = SupervisedDataset(
            list_data_dict=raw_datasets,
            tokenizer=tokenizer,
            template_variation=template_variation,
            context_length=context_length,
        )
    else:  # pre-tokenised (LESS) formats: prompt/completion or messages
        lm_datasets = encode_data(raw_datasets, tokenizer, context_length)

    return lm_datasets


def load_raw_dataset(
    train_files: list[str] | str,
    sample_size=None,
    sample_percentage=1.0,
    subset_index_files=None,
    seed=0,
    cache_dir=None,
    subset_selection="use_small_sources",
):
    """The raw rows of the files, or a subset of `sample_percentage` of them chosen by `subset_selection`."""
    if isinstance(train_files, str):
        train_files = [train_files]
    if len(train_files) == 1 and not train_files[0].endswith(".jsonl"):
        processed_datasets = load_dataset(train_files[0], cache_dir=cache_dir)["train"]
        if (subset_index_files is not None) and (len(subset_index_files) == 1):
            subset_indices = torch.load(subset_index_files[0], weights_only=True)
            processed_datasets = processed_datasets.select(subset_indices)
    else:
        processed_datasets = load_dataset(
            "json",
            data_files=train_files,
        )["train"]
        if (subset_index_files is not None) and (len(subset_index_files) == 1):
            subset_indices = torch.load(subset_index_files[0], weights_only=True)
            processed_datasets = processed_datasets.select(subset_indices)

    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    if subset_selection == "random":
        with utils.temp_seed(seed):
            index = np.random.permutation(len(processed_datasets))[:sample_size]

        sampled_dataset = processed_datasets.select(index)
        assert len(sampled_dataset) == sample_size
    elif subset_selection == "balanced_longest_selection":
        # Group examples by source
        source_groups = defaultdict(list)
        for idx, example in enumerate(processed_datasets):
            source_groups[example["source"]].append((idx, len(example["output"])))

        # Sort examples within each source by output length (descending)
        for source in source_groups:
            source_groups[source].sort(key=lambda x: x[1], reverse=True)

        # Calculate how many samples to take from each source
        num_sources = len(source_groups)
        samples_per_source = sample_size // num_sources
        extra_samples = sample_size % num_sources

        # Select longest samples from each source
        selected_indices = []
        for i, (_source, examples) in enumerate(source_groups.items()):
            num_to_select = samples_per_source + (1 if i < extra_samples else 0)
            selected_indices.extend([idx for idx, _ in examples[:num_to_select]])

        # If we don't have enough samples, take whatever is available
        if len(selected_indices) < sample_size:
            remaining = sample_size - len(selected_indices)
            all_remaining = [
                idx for source in source_groups.values() for idx, _ in source[samples_per_source:]
            ]
            selected_indices.extend(all_remaining[:remaining])

        # Shuffle the selected indices
        with utils.temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        assert len(sampled_dataset) == sample_size
    elif subset_selection == "longest_sourcewise_selection":
        # Group examples by source
        source_groups = defaultdict(list)
        for idx, example in enumerate(processed_datasets):
            source_groups[example["source"]].append((idx, len(example["output"])))

        # Sort examples within each source by output length (descending)
        for source in source_groups:
            source_groups[source].sort(key=lambda x: x[1], reverse=True)

        # Select longest samples from each source
        selected_indices = []
        for _source, examples in source_groups.items():
            # examples is a list of (idx, length) tuples
            selected_indices.extend(
                [idx for idx, _ in examples[: int(len(examples) * sample_percentage)]]
            )

        # Shuffle the selected indices
        with utils.temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        logger.info(f"Sampled dataset is len {len(sampled_dataset)}, sample size was {sample_size}")
        assert abs(len(sampled_dataset) - sample_size) <= 10
    elif subset_selection == "longest_selection":
        example_indices_and_lengths = [
            (idx, len(example["output"])) for idx, example in enumerate(processed_datasets)
        ]
        example_indices_and_lengths.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in example_indices_and_lengths[:sample_size]]

        # Shuffle the selected indices
        with utils.temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        assert len(sampled_dataset) == sample_size
    elif subset_selection == "use_small_sources":
        # Group examples by source
        source_groups = defaultdict(list)
        for idx, example in enumerate(processed_datasets):
            source_groups[example["source"]].append((idx, len(example["output"])))

        # Sort sources by size
        sorted_sources = sorted(source_groups.items(), key=lambda x: len(x[1]))
        small_sources = sorted_sources[:10]  # Select smallest 10 sources to use fully
        large_sources = sorted_sources[10:]  # Use some X% of the remaining sources

        # Calculate number of samples to take from large sources
        small_sources_total = sum(len(group) for _, group in small_sources)
        large_sources_sample_size = sample_size - small_sources_total
        large_sources_total = sum(len(group) for _, group in large_sources)
        large_source_percentage = large_sources_sample_size / large_sources_total

        selected_indices = []

        # Take all samples from small sources
        for _, group in small_sources:
            selected_indices.extend([idx for idx, _ in group])

        # Sort data in each of the large sources by output length and take the longest large_source_percentage% of samples
        for _, group in large_sources:
            group.sort(key=lambda x: x[1], reverse=True)
            group_sample_size = int(len(group) * large_source_percentage)
            group_selected_indices = [idx for idx, _ in group[:group_sample_size]]

            selected_indices.extend(group_selected_indices)

        # Shuffle selected indices
        with utils.temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        logger.info(
            f"Using small sources fully and {int(large_source_percentage * 100)}% of large sources"
        )
        assert abs(len(sampled_dataset) - sample_size) <= 5

    return sampled_dataset


def encode_data(raw_datasets, tokenizer, context_length: int, processing_num_workers=10):
    """Tokenise the `prompt`/`completion` or `messages` rows (no truncation) and drop the examples
    above `context_length` tokens; the drops are counted and logged per source."""
    if "input_ids" not in raw_datasets.features:  # else already encoded
        encode_function = get_encode_function(raw_datasets, tokenizer)
        logger.info(f"Encode function: {encode_function.func.__name__}")
        # To speed up this part, we use multiprocessing.
        raw_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=processing_num_workers,
            load_from_cache_file=False,
            desc="Tokenizing and reformatting instruction data",
        )
    lm_datasets = drop_too_long(raw_datasets, context_length)
    lm_datasets.set_format(type="pt")
    return lm_datasets


def drop_too_long(encoded_datasets, context_length: int):
    """The rows of an encoded dataset with at most `context_length` tokens (never truncated)."""
    lengths = np.array([len(ids) for ids in encoded_datasets["input_ids"]])
    fits = lengths <= context_length
    columns = encoded_datasets.column_names
    source_column = next((c for c in ("dataset", "source") if c in columns), None)
    names = encoded_datasets[source_column] if source_column else ["all"] * len(lengths)
    dropped = Counter(name for name, fit in zip(names, fits, strict=True) if not fit)
    logger.info(
        f"Dropped {sum(dropped.values())} of {len(lengths)} examples longer than "
        f"{context_length} tokens (never truncated): {dict(dropped)}"
    )
    return encoded_datasets.select(np.flatnonzero(fits))


def get_encode_function(raw_datasets, tokenizer):
    """The encode function of the columns of the dataset."""
    if "prompt" in raw_datasets.column_names and "completion" in raw_datasets.column_names:
        return partial(encode_with_prompt_completion_format, tokenizer=tokenizer)
    if "messages" in raw_datasets.column_names:
        return partial(encode_with_messages_format, tokenizer=tokenizer)
    raise ValueError(
        "You need to have either 'prompt'&'completion' or 'messages' in your column names."
    )


def encode_with_prompt_completion_format(example, tokenizer):
    """
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields. The prompt and the
    completion (+ EOS) are tokenised separately and concatenated, so that the tokens of the prompt
    are exactly those of the prompt alone, as at inference; the prompt is masked from the loss.
    Nothing is truncated.
    """
    prompt = tokenizer(example["prompt"], verbose=False)["input_ids"]
    completion = tokenizer(
        example["completion"] + tokenizer.eos_token, add_special_tokens=False, verbose=False
    )["input_ids"]
    input_ids = torch.tensor(prompt + completion)
    labels = input_ids.clone()
    labels[: len(prompt)] = IGNORE_INDEX  # mask the prompt part for avoiding loss
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
    }


def encode_with_messages_format(example, tokenizer):
    """
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together (the
    boundaries of the messages are found by tokenising the text up to them). Only the assistant
    messages are labels. Used for LESS datasets. Nothing is truncated.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def n_tokens(text):
        return len(tokenizer(text, verbose=False)["input_ids"])

    input_ids = torch.tensor(
        tokenizer(concat_messages(messages, tokenizer), verbose=False)["input_ids"]
    )
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = n_tokens(concat_messages(messages[:message_idx], tokenizer))
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = (
                    concat_messages(messages[: message_idx + 1], tokenizer) + "<|assistant|>\n"
                )
            else:
                messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer)
            labels[message_start_idx : n_tokens(messages_so_far)] = IGNORE_INDEX

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
    }


class SupervisedDataset(Dataset):
    """Prompt / completion pairs with their source, original index and completion length."""

    def __init__(
        self,
        list_data_dict: datasets.arrow_dataset.Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        template_variation: bool,
        context_length: int,
    ):
        """Examples whose prompt + completion (+ EOS) have more than `context_length` tokens are
        dropped (counted and logged per source), never truncated."""
        super().__init__()
        prompts = (
            utils.PROMPT_TEMPLATE[random.randrange(len(utils.PROMPT_TEMPLATE))]
            if template_variation
            else utils.PROMPT_TEMPLATE_SINGLE
        )
        prompt_input, prompt_no_input = prompts["prompt_input"], prompts["prompt_no_input"]

        self.sources, self.targets, names, self.indices, self.completion_lengths = (
            [],
            [],
            [],
            [],
            [],
        )
        discarded = 0
        for example in list_data_dict:
            # An example with an empty output (https://github.com/TIGER-AI-Lab/MAmmoTH/issues/36).
            if len(example["output"]) == 0:
                discarded += 1
                continue
            template = prompt_input if example.get("input", "") != "" else prompt_no_input
            self.sources.append(template.format_map(example))
            self.targets.append(f"{example['output']}{tokenizer.eos_token}")
            names.append(example["source"])
            self.indices.append(example.get("original_index", -1))
            self.completion_lengths.append(example.get("completion_length", -1))
        logger.info(f"Discarded {discarded} examples with an empty output")
        self._drop_too_long(tokenizer, names, context_length)

        # Data source names as integers, in sorted order.
        self.all_data_sources = sorted(set(names))
        ids = {name: i for i, name in enumerate(self.all_data_sources)}
        logger.info(f"Data sources: {ids}")
        self.data_sources = [ids[name] for name in names]
        self.num_sources = len(ids)

    def _drop_too_long(self, tokenizer, names, context_length: int) -> None:
        """Keep the examples that fit in `context_length` tokens (and have a completion to learn)."""
        fits, lengths = [], []
        # The tokenizer's own threads (train.py turns them off for the data loader workers).
        parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        for start in range(0, len(self.sources), 2048):
            prompts = tokenizer(self.sources[start : start + 2048], verbose=False)["input_ids"]
            completions = tokenizer(
                self.targets[start : start + 2048], add_special_tokens=False, verbose=False
            )["input_ids"]
            lengths += [len(p) + len(c) for p, c in zip(prompts, completions, strict=True)]
            fits += [
                0 < len(c) and len(p) + len(c) <= context_length
                for p, c in zip(prompts, completions, strict=True)
            ]
        os.environ["TOKENIZERS_PARALLELISM"] = parallelism or "false"
        dropped = Counter(name for name, fit in zip(names, fits) if not fit)
        logger.info(
            f"Dropped {sum(dropped.values())} of {len(fits)} examples longer than {context_length} "
            f"tokens (never truncated): {dict(dropped)}"
        )
        for values in (self.sources, self.targets, names, self.indices, self.completion_lengths):
            values[:] = [v for v, fit in zip(values, fits, strict=True) if fit]
        kept = [n for n, fit in zip(lengths, fits, strict=True) if fit]
        if not kept:
            raise ValueError(f"no example fits the context window of {context_length} tokens")
        self.mean_tokens = sum(kept) / len(kept)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        return dict(
            input_ids=self.sources[i],
            labels=self.targets[i],
            sources=self.data_sources[i],
            indices=self.indices[i],
            completion_lengths=self.completion_lengths[i],
        )


@dataclass
class SupervisedCollator:
    """Right-padded batch of prompt / completion strings (evaluation loss): `input_ids`, `labels`
    (-100 on the prompt), `attention_mask`; prompt and completion are tokenised separately and
    nothing is truncated."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict:
        examples = tokenize_examples(self.tokenizer, instances)
        pad = self.tokenizer.pad_token_id
        ids = [torch.from_numpy(e.input_ids) for e in examples]
        input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad)
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.from_numpy(e.labels) for e in examples],
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        lengths = torch.tensor([len(x) for x in ids])
        attention_mask = (torch.arange(input_ids.shape[1]) < lengths[:, None]).long()
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def tokenize_examples(tokenizer, instances: Sequence[dict]) -> list[Example]:
    """Prompt and completion tokenised separately and concatenated: the tokens of the prompt are
    exactly those of the prompt alone, as at inference, and nothing is truncated."""
    prompts = tokenizer([i["input_ids"] for i in instances])["input_ids"]
    completions = tokenizer([i["labels"] for i in instances], add_special_tokens=False)["input_ids"]
    return [
        Example(
            input_ids=np.array(p + c),
            labels=np.array([IGNORE_INDEX] * len(p) + c),
            source=i["sources"],
            index=i["indices"],
            completion_length=i["completion_lengths"],
        )
        for i, p, c in zip(instances, prompts, completions, strict=True)
    ]


@dataclass
class ExampleCollator:
    """The selection pool of a step as tokenised examples (packed later, per forward)."""

    make_examples: Callable[[Sequence[dict]], list[Example]]

    def __call__(self, instances: Sequence[dict]) -> dict:
        examples = self.make_examples(instances)
        return {"lengths": torch.tensor([len(e) for e in examples]), "examples": examples}


@dataclass
class PackCollator:
    """One packed batch (`colm.selection.packing`): the batches of the full-batch baseline."""

    make_examples: Callable[[Sequence[dict]], list[Example]]

    def __call__(self, instances: Sequence[dict]) -> dict:
        return pack(self.make_examples(instances))


def make_collator(args, tokenizer):
    """The collator of a MathInstruct-style run."""
    make = partial(tokenize_examples, tokenizer)
    return ExampleCollator(make) if args.coreset else PackCollator(make)


def concat_messages(messages, tokenizer):
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += (
                "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            )
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))

    return message_text
