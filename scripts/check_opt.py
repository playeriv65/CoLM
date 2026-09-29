#!/usr/bin/env python
"""Teacher-forced GPU check of the execution-only step optimisations against the plain path.

    python -u scripts/check_opt.py <config.json> <out.json> [key=value ...] [--steps N]

Runs `colm.train.train.main` on the optimised trainer (training continues on its own selections).
At every step, on the same weights, pool and selection state, it computes

* `ref`: the plain path: the features of every example of the pool (packs of `select_tokens`,
  the extractor called pack by pack, the features expanded from g_i), then the selector;
* `floor`: the plain path with the pool in reverse order: other packs, the same math, other
  fp32 rounding. This is how much the selection moves by rounding alone (the noise floor);
* `new`: the optimised `_select` (what the run continues with).

and records the selected dataset indices of each arm, the relative differences of g_i, and how
many examples were forwarded. For the first `--steps` steps it also compares the training
gradient of the examples `new` selected, dropout off, in fp16 autocast: the plain packs of one
micro-batch (twice: the GPU's own repeat noise), the optimised packs (`train_batches`), and an
fp32 reference without autocast; `summary` aggregates all of it.
"""

import argparse
import contextlib
import copy
import json
import os
import tempfile

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def _parse(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def install(record: dict, grad_steps: int):
    import colm.train.trainers as trainers
    from colm.selection.packing import greedy_groups, pack
    from colm.selection.pool import source_of

    cls = trainers.CoresetTrainer
    orig_select = cls._select

    def plain(self, examples, state):
        """Features of every example, pack by pack, and the selection: the former path."""
        self.model.eval()
        packs = self.batching.feature_batches(examples)
        g = torch.cat([self.extractor.extract(self._prepare_inputs(p)) for p in packs]).float()
        selector = copy.copy(self.selector)
        selector.prev_m, selector.prev_v = state
        total = self._per_rank(self.args.pool_micro_batches) * self.args.world_size
        chosen = selector(
            self.extractor.expand(g),
            [source_of(e) for e in examples],
            total,
            self.state.global_step,
        )
        return g.cpu(), chosen, selector

    def select(self, inputs):
        step = self.state.global_step
        examples = inputs["examples"]
        state = (self.selector.prev_m, self.selector.prev_v)
        g_ref, ref, ref_sel = plain(self, examples, state)
        g_floor, floor, _ = plain(self, examples[::-1], state)
        counted = []
        extract = self.extractor.extract
        self.extractor.extract = lambda p: counted.append(len(p["cu_seq_lens_q"]) - 1) or extract(p)
        sub_batches, total = orig_select(self, inputs)
        self.extractor.extract = extract
        idx = lambda sel, ex: [ex[i].index for i in sel.indices]  # noqa: E731
        selected = {
            "ref": idx(ref, examples),
            "floor": idx(floor, examples[::-1]),
            "new": [i for b, _ in sub_batches for i in b["colm_meta"]["indices"].tolist()],
        }
        entry = {
            "step": step,
            "selected": selected,
            "forwarded": sum(counted),
            "pool": len(examples),
            "packs": len(sub_batches),
            "train_tokens": [int(b["cu_seq_lens_q"][-1]) for b, _ in sub_batches],
            "new_equals_ref": sorted(selected["new"]) == sorted(selected["ref"]),
            "floor_equals_ref": sorted(selected["floor"]) == sorted(selected["ref"]),
        }
        g_floor = g_floor.flip(0)
        rel = ((g_floor - g_ref).abs() / g_ref.abs().clamp_min(1e-30)).numpy()
        entry["g_rel_floor"] = {"median": float(np.median(rel)), "max": float(rel.max())}
        if self.selector.prev_m is not None and ref_sel.prev_m is not None:
            entry["adam_m_rel"] = _rel(self.selector.prev_m, ref_sel.prev_m)
        if step < grad_steps:
            entry.update(gradient_check(self, examples, sub_batches, total))
        record["steps"].append(entry)
        print(
            f"[check] {json.dumps({k: v for k, v in entry.items() if k != 'selected'})}", flush=True
        )
        return sub_batches, total

    def gradient_check(self, pool, sub_batches, total):
        """Gradients of the selected examples: plain micro-batch packs vs the optimised packs."""
        model = self.model
        selected = [e for b, _ in sub_batches for e in _examples(pool, b)]
        weights = [1.0] * len(selected)
        micro = int(
            np.mean([len(e) for e in pool])
            * self.args.micro_batch_size
            * self.args.small_batch_ratio
        )
        params = [p for p in model.parameters() if p.requires_grad]
        rng = torch.cuda.get_rng_state()

        def run(groups, amp):
            model.zero_grad(set_to_none=True)
            model.eval()  # no dropout: comparable gradients
            saved = model.forward
            if not amp:  # accelerate wraps the forward in autocast: put the plain one back
                model.forward = model._original_forward
            loss_value = 0.0
            # The fp32 reference uses the exact MATH attention: the fp32 memory-efficient kernel
            # is ~0.25 (relative) off in the gradients of phi-2 and depends on the packing.
            exact = contextlib.nullcontext() if amp else sdpa_kernel(SDPBackend.MATH)
            try:
                with exact:
                    for group in groups:
                        batch = self._prepare_inputs(pack([selected[i] for i in group]))
                        w = torch.tensor([weights[i] for i in group])
                        loss = self.batching.loss(self, model, batch, w, total)
                        (loss * 1024).backward()
                        loss_value += float(loss.detach())
            finally:
                model.forward = saved
            grads = torch.cat(
                [p.grad.double().flatten() / 1024 for p in params if p.grad is not None]
            )
            return loss_value, grads

        plain_groups = greedy_groups([len(e) for e in selected], micro)
        new_groups = greedy_groups([len(e) for e in selected], self.batching.train_tokens)
        loss_a, g_a = run(plain_groups, True)
        loss_b, g_b = run(plain_groups, True)
        loss_n, g_n = run(new_groups, True)
        loss_32, g_32 = run(plain_groups, False)
        loss_n32, g_n32 = run(new_groups, False)
        model.zero_grad(set_to_none=True)
        torch.cuda.set_rng_state(rng)

        def cos(a, b):
            return float(a @ b / (a.norm() * b.norm()))

        return {
            "grad_packs": [len(plain_groups), len(new_groups)],
            "loss_rel_new": abs(loss_n - loss_a) / abs(loss_a),
            "loss_rel_repeat": abs(loss_b - loss_a) / abs(loss_a),
            "loss_rel_fp32": abs(loss_a - loss_32) / abs(loss_32),
            "loss_rel_packs_fp32": abs(loss_n32 - loss_32) / abs(loss_32),
            "grad_rel_new": _rel(g_n, g_a),
            "grad_rel_repeat": _rel(g_b, g_a),
            "grad_rel_packs_fp32": _rel(g_n32, g_32),  # the packing alone, in fp32
            "grad_rel_plain_vs_fp32": _rel(g_a, g_32),
            "grad_rel_new_vs_fp32": _rel(g_n, g_32),
            "grad_cos_plain_vs_fp32": cos(g_a, g_32),
            "grad_cos_new_vs_fp32": cos(g_n, g_32),
            "grad_norm_plain": float(g_a.norm()),
            "grad_norm_new": float(g_n.norm()),
        }

    def _examples(pool, batch):
        by_index = {e.index: e for e in pool}
        return [by_index[i] for i in batch["colm_meta"]["indices"].tolist()]

    cls._select = select
    import transformers

    transformers.Trainer.save_model = lambda self, *a, **k: None


def summarize(record: dict) -> dict:
    steps = record["steps"]
    grad = [s for s in steps if "grad_rel_new" in s]
    overlap = lambda a: float(  # noqa: E731
        np.mean([len(set(s["selected"][a]) & set(s["selected"]["ref"])) for s in steps])
    )
    out = {
        "steps": len(steps),
        "selected_per_step": len(steps[0]["selected"]["ref"]),
        "new_equals_ref": sum(s["new_equals_ref"] for s in steps),
        "floor_equals_ref": sum(s["floor_equals_ref"] for s in steps),
        "overlap_new": overlap("new"),
        "overlap_floor": overlap("floor"),
        "g_rel_floor_median": float(np.median([s["g_rel_floor"]["median"] for s in steps])),
        "forwarded_fraction": sum(s["forwarded"] for s in steps) / sum(s["pool"] for s in steps),
        "adam_m_rel_max": max((s["adam_m_rel"] for s in steps if "adam_m_rel" in s), default=None),
    }
    for key in (
        "loss_rel_new",
        "loss_rel_repeat",
        "loss_rel_fp32",
        "loss_rel_packs_fp32",
        "grad_rel_new",
        "grad_rel_packs_fp32",
        "grad_rel_repeat",
        "grad_rel_plain_vs_fp32",
        "grad_rel_new_vs_fp32",
        "grad_cos_plain_vs_fp32",
        "grad_cos_new_vs_fp32",
    ):
        if grad:
            out[f"{key}_mean"] = float(np.mean([s[key] for s in grad]))
            out[f"{key}_max"] = float(np.max([s[key] for s in grad]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("out")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--steps", type=int, default=10**9, help="steps with a gradient check")
    ns = parser.parse_args()
    with open(ns.config) as f:
        config = json.load(f)
    for item in ns.overrides:
        key, value = item.split("=", 1)
        config[key] = _parse(value)
    config["profile_timing"] = "off"
    record = {"config": config, "steps": []}
    install(record, ns.steps)
    out_dir = os.path.dirname(os.path.abspath(ns.out))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", dir=out_dir, prefix="check-config-", delete=False
    ) as f:
        json.dump(config, f, indent=1)
        merged = f.name
    from colm.train import train

    try:
        train.main([merged])
    finally:
        os.remove(merged)
        record["summary"] = summarize(record) if record["steps"] else {}
        with open(ns.out, "w") as f:
            json.dump(record, f, indent=1)
        print(f"[check] summary {json.dumps(record['summary'])}")
        print(f"check -> {ns.out}")


if __name__ == "__main__":
    main()
