import contextlib
import logging
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import datasets
import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset

import colm.data.utils as utils

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def get_training_dataset(
    train_files: list[str],
    tokenizer,
    max_seq_length,
    sample_percentage=1.0,
    subset_index_files=None,
    template_variation=False,
    seed=0,
    hf_datasets_cache_dir=None,
):
    """get training dataset with a specified seed"""
    raw_datasets = load_raw_dataset(
        train_files,
        sample_percentage=sample_percentage,
        subset_index_files=subset_index_files,
        seed=seed,
        cache_dir=hf_datasets_cache_dir,
    )

    if "instruction" in raw_datasets.column_names:
        lm_datasets = SupervisedDataset(
            list_data_dict=raw_datasets, tokenizer=tokenizer, template_variation=template_variation
        )
    else:
        lm_datasets = encode_data(raw_datasets, tokenizer, max_seq_length)

    return lm_datasets


def load_raw_dataset(
    train_files: list[str] | str,
    sample_size=None,
    sample_percentage=1.0,
    subset_index_files=None,
    seed=0,
    cache_dir=None,
):
    """load raw dataset"""
    if isinstance(train_files, str):
        train_files = [train_files]
    if len(train_files) == 1 and not train_files[0].endswith(".jsonl"):
        processed_datasets = load_dataset(train_files[0], cache_dir=cache_dir)["train"]
        if (subset_index_files is not None) and (len(subset_index_files) == 1):
            subset_indices = torch.load(subset_index_files[0])
            processed_datasets = processed_datasets.select(subset_indices)
    else:
        processed_datasets = load_dataset(
            "json",
            data_files=train_files,
        )["train"]
        if (subset_index_files is not None) and (len(subset_index_files) == 1):
            subset_indices = torch.load(subset_index_files[0])
            processed_datasets = processed_datasets.select(subset_indices)

    print(f"Before selection, keys are {processed_datasets[0].keys()}")

    if sample_size is None:
        sample_size = int(len(processed_datasets) * sample_percentage)

    if sample_size == len(processed_datasets):
        return processed_datasets  # not shuffle

    subset_selection = "use_small_sources"
    if subset_selection == "random":
        with temp_seed(seed):
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
        with temp_seed(seed):
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
        with temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        print(f"Sampled dataset is len {len(sampled_dataset)}, sample size was {sample_size}")
        assert abs(len(sampled_dataset) - sample_size) <= 10
    elif subset_selection == "longest_selection":
        example_indices_and_lengths = [
            (idx, len(example["output"])) for idx, example in enumerate(processed_datasets)
        ]
        example_indices_and_lengths.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [idx for idx, _ in example_indices_and_lengths[:sample_size]]

        # Shuffle the selected indices
        with temp_seed(seed):
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
        with temp_seed(seed):
            np.random.shuffle(selected_indices)

        sampled_dataset = processed_datasets.select(selected_indices)
        print(
            f"Using small sources fully and {int(large_source_percentage * 100)}% of large sources"
        )
        assert abs(len(sampled_dataset) - sample_size) <= 5

    return sampled_dataset


def encode_data(
    raw_datasets,
    tokenizer,
    max_seq_length,
    processing_num_workers=10,
    overwrite_cache=False,
    func_name="encode_with_messages_format",
):
    """encode data with the specified tokenizer and the chat format."""
    # if already encoded, return
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    encode_function = get_encode_function(raw_datasets, tokenizer, max_seq_length, func_name)
    print(f"USING ENCODE FUNCTION {encode_function}")
    # To speed up this part, we use multiprocessing.
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")

    return lm_datasets


def get_encode_function(
    raw_datasets, tokenizer, max_seq_length, func="encode_with_messages_format"
):
    """get encode function based on the dataset."""
    if "prompt" in raw_datasets.column_names and "completion" in raw_datasets.column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    elif "messages" in raw_datasets.column_names:
        if func == "encode_with_messages_format":
            encode_func = encode_with_messages_format
        else:
            encode_func = encode_with_messages_format_with_llama2_chat
        encode_function = partial(
            encode_func,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    else:
        raise ValueError(
            "You need to have either 'prompt'&'completion' or 'messages' in your column names."
        )
    return encode_function


def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    """
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L238

    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    """
    # if prompt doesn't end with space and completion doesn't start with space, add space
    if not example["prompt"].endswith((" ", "\n", "\t")) and not example["completion"].startswith(
        (" ", "\n", "\t")
    ):
        example_text = example["prompt"] + " " + example["completion"]
    else:
        example_text = example["prompt"] + example["completion"]
    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(
        example["prompt"], return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    # mask the prompt part for avoiding loss
    labels[:, : tokenized_prompt.input_ids.shape[1]] = IGNORE_INDEX
    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Original implementation of the function: https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L264C1-L322C1

    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    Used for LESS datasets.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = (
                    concat_messages(messages[: message_idx + 1], tokenizer) + "<|assistant|>\n"
                )
            else:
                messages_so_far = concat_messages(messages[: message_idx + 1], tokenizer)
            message_end_idx = tokenizer(
                messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = IGNORE_INDEX

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


class SupervisedDataset(Dataset):
    """Prompt / completion pairs with their source, original index and completion length."""

    def __init__(
        self,
        list_data_dict: datasets.arrow_dataset.Dataset,
        tokenizer: transformers.PreTrainedTokenizer,
        template_variation: bool,
    ):
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

        # Data source names as integers, in sorted order.
        self.all_data_sources = sorted(set(names))
        ids = {name: i for i, name in enumerate(self.all_data_sources)}
        logger.info(f"Data sources: {ids}")
        self.data_sources = [ids[name] for name in names]
        self.num_sources = len(ids)

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
    """Tokenise prompt / completion strings and pad them into a batch.

    The batch carries `input_ids`, `labels` (-100 on the prompt), `attention_mask` and the
    per-example `colm_meta` (source, original index, completion length).
    `legacy` (upstream errors E4a, E15): the prompt and the completion are tokenised together, so
    a token can straddle their boundary (9% of the examples) and the prompt is masked by the
    length of its separate tokenisation, and the attention mask is `input_ids != pad_token_id`.
    """

    tokenizer: transformers.PreTrainedTokenizer
    legacy: bool = False

    def _tokenize(self, prompts, completions):
        tok, max_length = self.tokenizer, self.tokenizer.model_max_length
        if self.legacy:
            joint = tok(
                [p + c for p, c in zip(prompts, completions)],
                truncation=True,
                max_length=max_length,
            )
            prompt_ids = tok(prompts, truncation=True, max_length=max_length)["input_ids"]
            ids = [torch.tensor(x) for x in joint["input_ids"]]
            labels = [x.clone() for x in ids]
            for label, prompt in zip(labels, prompt_ids):
                label[: len(prompt)] = IGNORE_INDEX
            return ids, labels
        prompt_ids = tok(prompts)["input_ids"]
        completion_ids = tok(completions, add_special_tokens=False)["input_ids"]
        ids, labels = [], []
        for p, c in zip(prompt_ids, completion_ids):
            ids.append(torch.tensor((p + c)[:max_length]))
            labels.append(torch.tensor(([IGNORE_INDEX] * len(p) + c)[:max_length]))
        return ids, labels

    def __call__(self, instances: Sequence[dict]) -> dict:
        ids, labels = self._tokenize(
            [i["input_ids"] for i in instances], [i["labels"] for i in instances]
        )
        pad = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        if self.legacy:
            attention_mask = input_ids.ne(pad)
        else:
            lengths = torch.tensor([len(x) for x in ids])
            attention_mask = (torch.arange(input_ids.shape[1]) < lengths[:, None]).long()
        meta = {
            "sources": torch.tensor([i["sources"] for i in instances]),
            "indices": torch.tensor([i["indices"] for i in instances]),
            "completion_lengths": torch.tensor([i["completion_lengths"] for i in instances]),
        }
        return dict(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, colm_meta=meta
        )


def make_collator(args, tokenizer):
    """The collator of a MathInstruct-style run: pools for coreset training, batches otherwise."""
    collate = SupervisedCollator(tokenizer, legacy=args.legacy)
    return PoolCollator(collate, args.micro_batch_size) if args.coreset else collate


@dataclass
class PoolCollator:
    """One selection pool (all examples of an optimizer step) as a list of micro-batches."""

    collate: SupervisedCollator
    micro_batch_size: int

    def __call__(self, instances: Sequence[dict]) -> dict:
        n = self.micro_batch_size
        return {
            "micro_batches": [
                self.collate(instances[i : i + n]) for i in range(0, len(instances), n)
            ]
        }


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


def encode_with_messages_format_with_llama2_chat(example, tokenizer, max_seq_length):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(
        messages,
    ):
        B_INST, E_INST = "[INST]", "[/INST]"
        bos = "<s>"
        eos = "</s>"
        formatted_text = ""

        for message in messages:
            if message["role"] == "user":
                formatted_text += bos + f"{B_INST} {(message['content']).strip()} {E_INST}"
            elif message["role"] == "assistant":
                formatted_text += f" {(message['content'])} " + eos
            else:
                raise ValueError(
                    "Llama2 chat template only supports 'system', 'user' and 'assistant' roles. Invalid role: {}.".format(
                        message["role"]
                    )
                )
        formatted_text = formatted_text[len(bos) :]

        return formatted_text

    example_text = _concat_messages(messages).strip()
    print(example_text)
    tokenized_example = tokenizer(
        example_text, return_tensors="pt", max_length=max_seq_length, truncation=True
    )
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if messages[message_idx + 1]["role"] == "assistant":
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = IGNORE_INDEX

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }
