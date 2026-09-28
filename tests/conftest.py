"""Tiny random Phi + character tokenizer + synthetic data-mixture fixtures (CPU only)."""

import json
import os
import string

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PhiConfig, PhiForCausalLM, PreTrainedTokenizerFast

SPECIAL_TOKENS = ["<unk>", "</s>", "<pad>"]
NUM_LAYERS = 2
NUM_SOURCES = 4


@pytest.fixture(scope="session")
def tokenizer():
    chars = sorted(set(string.printable))
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS + chars)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=512,
    )
    return tok


def make_phi(tokenizer, seed=0):
    torch.manual_seed(seed)
    config = PhiConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        max_position_embeddings=512,
        partial_rotary_factor=0.5,
        pad_token_id=tokenizer.pad_token_id,
    )
    config._attn_implementation = "sdpa"
    return PhiForCausalLM(config)


def add_lora(model, r=4):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    return model


@pytest.fixture()
def phi(tokenizer):
    return make_phi(tokenizer)


@pytest.fixture()
def mixture_file(tmp_path):
    """jsonl in the MathInstruct format: instruction/input/output/source."""
    rows = []
    for i in range(64):
        source = f"source_{i % NUM_SOURCES}"
        rows.append(
            {
                "instruction": f"Add {i} and {i % 7}.",
                "input": "" if i % 2 else f"numbers {i} {i % 7}",
                "output": f"The answer is {i + i % 7}." + " ok" * (i % 5),
                "source": source,
                "original_index": i,
            }
        )
    path = tmp_path / "mixture.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)
