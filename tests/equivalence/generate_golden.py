"""Generate the golden files of the pre-refactor code (tag `pre-refactor`).

Run ONLY against a checkout of the tag (the reference), from that checkout's root:

    cd <checkout of pre-refactor> && uv run python <path>/tests/equivalence/generate_golden.py --out <dir>

Everything is CPU and float64: with fp32 the MeZO feature is decided at rounding level (docs F8),
so float64 identity is the exactness criterion. The reference is patched in two places that only
matter for float64 (`logits.float()` in the decomposed forward and the identical cast in the
feature gather); for fp32 / fp16 models both patches are no-ops.
"""

import argparse
import json
import os
import string
import sys
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import PhiConfig, PhiForCausalLM, PreTrainedTokenizerFast

from colm.data.get_training_dataset import (
    DataCollatorForSupervisedDatasetWithSource,
    get_training_dataset,
)
from colm.train.facility_location import get_orders_and_weights
from colm.train.trainers import CustomTrainer, SubsetTrainer, SubsetTrainerEfficient
from colm.train.training_arguments import TrainingArguments

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


def model_fp64(tok, lora_dropout=0.0, resid_pdrop=0.0):
    torch.manual_seed(0)
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
    model = PhiForCausalLM(config)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=16,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"],
    )
    model = get_peft_model(model, lora)
    model.enable_input_require_grads()
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.05)
    return model.double()


def patch_reference():
    """float64 identity: keep the dtype of the logits (no-op for fp32/fp16 models)."""
    from colm.train import custom_phi

    def forward_final_layer(self, intermediate, labels=None, per_sample_loss=True):
        hidden_states = self.layers[-1](
            intermediate["hidden_states"],
            attention_mask=intermediate["causal_mask"],
            position_ids=intermediate["position_ids"],
            position_embeddings=intermediate["position_embeddings"],
        )
        hidden_states = self.transformer.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
        if labels is None:
            return None, logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        if per_sample_loss:
            b, s = shift_labels.shape
            per_token = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), reduction="none"
            ).view(b, s)
            return per_token.mean(dim=1), logits
        return (
            torch.nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            ),
            logits,
        )

    custom_phi.DecomposedPhiCausalLM.forward_final_layer = forward_final_layer


def args_for(out, **kw):
    base = dict(
        output_dir=str(out),
        use_cpu=True,
        max_steps=3,
        learning_rate=1e-3,
        warmup_steps=0,
        logging_steps=1,
        save_strategy="no",
        last_layer_index=NUM_LAYERS - 1,
        zo_dim=16,
        dataloader_num_workers=0,
    )
    base.update(kw)
    a = TrainingArguments(**base)
    a.keep_sources = [int(s) for s in a.keep_sources.split("_")] if a.keep_sources else []
    a.last_layers = [n + ".lora_B" for n in a.last_layers]
    return a


def build(cls, args, tok, data, lora_dropout=0.0):
    model = model_fp64(tok, lora_dropout=lora_dropout)
    ds = get_training_dataset([data], tokenizer=tok, max_seq_length=512)
    return cls(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tok,
        data_collator=DataCollatorForSupervisedDatasetWithSource(tokenizer=tok),
    ), model


def lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


# ---------------------------------------------------------------------------------------------
def gen_data(tok, data):
    ds = get_training_dataset([data], tokenizer=tok, max_seq_length=512)
    col = DataCollatorForSupervisedDatasetWithSource(tokenizer=tok)
    out = {"len": len(ds), "num_sources": ds.num_sources, "all_data_sources": ds.all_data_sources}
    batches = []
    for start in (0, 4, 8, 13):
        b = col([ds[i] for i in range(start, start + 4)])
        batches.append(
            {
                "input_ids": b["input_ids"],
                "labels": b["labels"],
                "attention_mask": b["attention_mask"].long(),
                "sources": [int(s) for s in b["sources"]],
                "indices": [int(s) for s in b["indices"]],
                "completion_lengths": [int(s) for s in b["completion_lengths"]],
            }
        )
    out["batches"] = batches
    return out


def gen_fl():
    rng = np.random.default_rng(0)
    cases = []
    ties = rng.integers(0, 4, size=(40, 6)).astype(np.float32)
    for name, X, y in [
        ("random", rng.normal(size=(40, 16)).astype(np.float32), None),
        ("ties", ties, None),
        ("sources", rng.normal(size=(48, 8)).astype(np.float32), np.array([0] * 24 + [5] * 16 + [9] * 8)),
    ]:
        for metric in ("l1", "euclidean", "cosine"):
            for strategy in ("none", "proportional"):
                if (y is None) != (strategy == "none"):
                    continue
                for start in ("floor", "ceil"):
                    order, w = get_orders_and_weights(
                        12, torch.from_numpy(X), metric, y=y, per_class_start=start, strategy=strategy
                    )
                    cases.append(
                        {"name": name, "X": torch.from_numpy(X), "y": None if y is None else torch.from_numpy(y),
                         "metric": metric, "strategy": strategy, "start": start,
                         "order": torch.from_numpy(order.astype(np.int64)), "weights": torch.from_numpy(w)}
                    )
    return cases


SELECT_GRID = [
    dict(),
    dict(mezo_transform="self_normalize"),
    dict(mezo_transform="normalize"),
    dict(mezo_transform="clip_full"),
    dict(mezo_transform="clip_last"),
    dict(mezo_topk="smallest"),
    dict(mezo_topk="largest_smallest"),
    dict(mezo_selection="weight"),
    dict(mezo_optim="sgd"),
    dict(facility_similarity="cosine"),
    dict(facility_similarity="euclidean", source_wise_selection="none"),
    dict(source_wise_selection="balanced", num_per_class_start="ceil"),
    dict(data_selection_method="weightedsubmodlib"),
    dict(keep_sources="0_2"),
    dict(keep_sources=""),
]


def gen_select(tok, data, tmp):
    """`_select_on_main` on synthetic features over a 5-step sequence, per configuration."""
    cases = []
    for ci, over in enumerate(SELECT_GRID):
        args = args_for(tmp / f"sel{ci}", per_device_train_batch_size=1, gradient_accumulation_steps=32,
                        small_batch_ratio=0.5, **over)
        trainer, model = build(SubsetTrainer, args, tok, data)
        ds = trainer.train_dataset
        rng = np.random.RandomState(100 + ci)
        d = trainer.param_dim
        steps = []
        for step in range(5):
            n = 32
            idx = rng.permutation(len(ds))[:n]
            feats = torch.from_numpy(rng.normal(size=(n, d)) * (0.5 + rng.rand(n, 1))).to(torch.float32)
            examples = [{"sources": [ds.data_sources[i]], "indices": [ds.indices[i]]} for i in idx]
            trainer.state.global_step = step
            torch.manual_seed(step)
            np.random.seed(step)
            sel, w = trainer._select_on_main(feats.clone(), examples, 16)
            steps.append({"feats": feats, "sources": [ds.data_sources[i] for i in idx], "step": step,
                          "selected": list(sel), "weights": list(w)})
        cases.append({"over": {k: v for k, v in over.items()}, "steps": steps,
                      "param_dim": d, "prev_m": trainer.prev_m_t, "prev_v": trainer.prev_v_t})
    return cases


def gen_features(tok, data, tmp):
    out = {}
    for unit in ("rep", "mezo", "masked_grad", "completion_length", "length_loss_weighted"):
        args = args_for(tmp / f"f{unit}", per_device_train_batch_size=1, gradient_accumulation_steps=4,
                        data_selection_unit=unit, keep_sources="")
        trainer, model = build(SubsetTrainer, args, tok, data)
        loader = trainer.get_train_dataloader()
        feats = []
        for i, batch in enumerate(loader):
            if i == 3:
                break
            torch.manual_seed(5)
            model.train()
            feats.append(trainer.save_select(batch))
        out[unit] = [torch.as_tensor(f) for f in feats]
        out[unit + "_param"] = lora_state(model)
    args = args_for(tmp / "feff", per_device_train_batch_size=4, gradient_accumulation_steps=2,
                    efficient_mezo=True, keep_sources="")
    trainer, model = build(SubsetTrainerEfficient, args, tok, data)
    loader = trainer.get_train_dataloader()
    out["efficient"] = [trainer.save_select(b) for _, b in zip(range(2), loader)]
    # per-sample losses of the decomposed forward on a padded batch
    dec = trainer.decomposer
    b = next(iter(loader))
    with torch.no_grad():
        mid = dec.forward_till_penultimate(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        loss, logits = dec.forward_final_layer(mid, labels=b["labels"], per_sample_loss=True)
    out["efficient_loss"] = loss
    out["efficient_param"] = lora_state(model)
    return out


TRAJ = {
    "custom": (CustomTrainer, dict(per_device_train_batch_size=2, gradient_accumulation_steps=2,
                                   data_selection_method="none")),
    "efficient": (SubsetTrainerEfficient, dict(per_device_train_batch_size=4, gradient_accumulation_steps=2,
                                               efficient_mezo=True, keep_sources="0")),
    "efficient_nokeep": (SubsetTrainerEfficient, dict(per_device_train_batch_size=4, gradient_accumulation_steps=2,
                                                      efficient_mezo=True, keep_sources="")),
    "efficient_dropout": (SubsetTrainerEfficient, dict(per_device_train_batch_size=4, gradient_accumulation_steps=2,
                                                       efficient_mezo=True, keep_sources="0")),
    "subset_mezo": (SubsetTrainer, dict(per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                        data_selection_unit="mezo", keep_sources="0")),
    "subset_mezo_weighted": (SubsetTrainer, dict(per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                                 data_selection_unit="mezo", keep_sources="",
                                                 data_selection_method="weightedsubmodlib")),
    "subset_rep": (SubsetTrainer, dict(per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                       data_selection_unit="rep", keep_sources="")),
    "subset_masked_grad": (SubsetTrainer, dict(per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                               data_selection_unit="masked_grad", keep_sources="")),
    "subset_completion_length": (SubsetTrainer, dict(per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                                     data_selection_unit="completion_length", keep_sources="")),
    "subset_length_loss": (SubsetTrainer, dict(per_device_train_batch_size=1, gradient_accumulation_steps=8,
                                               data_selection_unit="length_loss_weighted", keep_sources="")),
}


def gen_trajectories(tok, data, tmp):
    out = {}
    for name, (cls, over) in TRAJ.items():
        dropout = 0.2 if name.endswith("dropout") else 0.0
        args = args_for(tmp / name, **over)
        trainer, model = build(cls, args, tok, data, lora_dropout=dropout)
        selections, trained, rng_first = [], [], []
        if hasattr(trainer, "_select_on_main"):
            orig_sel = trainer._select_on_main

            def sel(all_reps, complete_examples, total, _o=orig_sel):
                idx, w = _o(all_reps, complete_examples, total)
                selections.append({"pool": [int(e["indices"][0]) for e in complete_examples],
                                   "selected": list(idx), "weights": list(w)})
                return idx, w

            trainer._select_on_main = sel
        orig_ts = trainer.training_step

        def ts(m, inputs, num=None, _o=orig_ts):
            if inputs:
                trained.append({"step": trainer.state.global_step, "indices": [int(i) for i in inputs["indices"]],
                                "weight": None if "colm_sample_weight" not in inputs else float(inputs["colm_sample_weight"])})
                if not rng_first or rng_first[-1][0] != trainer.state.global_step:
                    rng_first.append((trainer.state.global_step, torch.get_rng_state().clone()))
            return _o(m, inputs, num)

        trainer.training_step = ts
        trainer.train()
        out[name] = {
            "selections": selections,
            "trained": trained,
            "log": [{k: v for k, v in h.items() if k in ("step", "loss", "grad_norm", "learning_rate")}
                    for h in trainer.state.log_history if "loss" in h],
            "lora": lora_state(model),
            "rng_at_train_start": [s for _, s in rng_first],
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import pathlib
    import tempfile

    tmp = pathlib.Path(tempfile.mkdtemp())
    tok = tokenizer()
    data = str(tmp / "mixture.jsonl")
    mixture(data)
    patch_reference()
    parts = {
        "data": gen_data(tok, data),
        "fl": gen_fl(),
        "select": gen_select(tok, data, tmp),
        "features": gen_features(tok, data, tmp),
        "trajectories": gen_trajectories(tok, data, tmp),
    }
    for name, part in parts.items():
        torch.save(part, os.path.join(args.out, f"{name}.pt"))
        print("wrote", name, flush=True)


if __name__ == "__main__":
    main()
