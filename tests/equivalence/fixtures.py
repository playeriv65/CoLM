"""Tiny float64 Phi + character tokenizer + synthetic mixture shared by the golden generator and the tests.

Imports nothing from `colm`, so the generator can run it against the pre-refactor checkout.
"""

import json
import string

import torch
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PhiConfig, PhiForCausalLM, PreTrainedTokenizerFast

NUM_LAYERS, NUM_SOURCES = 2, 4


def tokenizer():
    chars = sorted(set(string.printable))
    vocab = {t: i for i, t in enumerate(["<unk>", "</s>", "<pad>"] + chars)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=512,
    )


def mixture(path):
    rows = []
    for i in range(64):
        rows.append(
            {
                "instruction": f"Add {i} and {i % 7}.",
                "input": "" if i % 2 else f"numbers {i} {i % 7}",
                "output": f"The answer is {i + i % 7}." + " ok" * (i % 5),
                "source": f"source_{i % NUM_SOURCES}",
                "original_index": i,
                "completion_length": 5 + i % 5,
            }
        )
    with open(path, "w") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")


def make_phi(tok, seed=0, resid_pdrop=0.0):
    torch.manual_seed(seed)
    config = PhiConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        max_position_embeddings=512,
        partial_rotary_factor=0.5,
        pad_token_id=tok.pad_token_id,
        resid_pdrop=resid_pdrop,
    )
    config._attn_implementation = "sdpa"
    return PhiForCausalLM(config)


def add_lora(model, r=4, lora_dropout=0.0):
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=16,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    return model


def model_fp64(tok, lora_dropout=0.0, resid_pdrop=0.0):
    """Tiny Phi with LoRA (B randomised, LoRA B starts at zero otherwise), in float64."""
    model = add_lora(make_phi(tok, 0, resid_pdrop), lora_dropout=lora_dropout)
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.05)
    return model.double()
