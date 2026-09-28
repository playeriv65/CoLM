#!/usr/bin/env python
"""Teacher-forced GPU check of the exact step optimisations against the original path.

    python -u scripts/check_exact_opts.py <config.json> <out.json> [key=value ...]

Runs `colm.train.train.main` (training continues on the configured, i.e. optimised, path).
At every step, on the same weights, batch, MeZO Adam state, perturbation and RNG state, the
selection is computed four times:

* `ref`: the original path (all selection flags off, per-micro-batch padded forward);
* `floor`: the original path with the rows of every micro-batch reversed. Same widths and
  divisors, so the same math; only fp32 rounding differs. This is the noise floor: how often
  plain fp32 reordering alone changes the selection;
* `rows`: the original path with every row as its own micro-batch (width and divisor kept):
  the same math with different GEMM shapes for every example, a floor comparable to packing;
* `new`: the configured path (the one that continues).

For the first `grad_steps` steps (default: all) it also computes the training gradient of the
examples `new` selected, with dropout off (eval mode) so that arms are comparable: fp16 AMP
original (padded sub-batches, the model's attention; twice, for the GPU nondeterminism floor),
fp16 AMP configured (packing / attention), and fp32 eager without autocast, padded and packed
(one row per sub-batch) — the fp32 pair isolates the packing math from the fp16 kernels and is
the reference the fp16 arms are measured against.

Per step it records the selected dataset indices of every arm, g_i relative differences, the
MeZO Adam state difference, whether the RNG state after selection is identical, and the loss /
gradient differences; `summary` aggregates them.
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch


def _parse(value: str):
    """JSON value (numbers, true/false/null, lists) or else the plain string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


SELECTION_FLAGS = (
    "lazy_mode_switch",
    "skip_unused_features",
    "zo_label_positions_only",
    "zo_packing",
)
TRAIN_FLAGS = ("train_packing",)


def _flip(batch: dict) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.flip(0) if isinstance(v, torch.Tensor) else list(reversed(v))
    return out


def _rows(batch: dict) -> list[dict]:
    """The rows of a collated batch as single-row batches (padding and width kept)."""
    return [{k: v[i : i + 1] for k, v in batch.items()} for i in range(len(batch["input_ids"]))]


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def install(record: dict, grad_steps: int):
    import colm.train.trainers as trainers

    cls = trainers.SubsetTrainerEfficient
    orig_select = cls._select_microbatches
    orig_on_main = cls._select_on_main

    def capture_on_main(self, all_reps, complete_examples, total):
        idx, weights = orig_on_main(self, all_reps, complete_examples, total)
        self._check_capture = {
            "reps": all_reps.float().cpu(),
            "ids": [int(ex["indices"][0]) for ex in complete_examples],
            "selected": [int(complete_examples[i]["indices"][0]) for i in idx],
        }
        return idx, weights

    cls._select_on_main = capture_on_main

    def state(self):
        param = self.named_parameters_to_optim[0][1]
        return {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "np": np.random.get_state(),
            "m": None if self.prev_m_t is None else self.prev_m_t.clone(),
            "v": None if self.prev_v_t is None else self.prev_v_t.clone(),
            "param": param.data.clone(),
            "flags": {k: getattr(self.args, k) for k in SELECTION_FLAGS + TRAIN_FLAGS},
            "new_path": self._zo_new_path,
        }

    def restore(self, s):
        torch.set_rng_state(s["cpu"])
        torch.cuda.set_rng_state_all(s["cuda"])
        np.random.set_state(s["np"])
        self.prev_m_t = None if s["m"] is None else s["m"].clone()
        self.prev_v_t = None if s["v"] is None else s["v"].clone()
        self.named_parameters_to_optim[0][1].data = s["param"].clone()
        for k, v in s["flags"].items():
            setattr(self.args, k, v)
        self._zo_new_path = s["new_path"]

    def original_flags(self):
        for k in SELECTION_FLAGS:
            setattr(self.args, k, False)
        self.args.train_packing = "none"
        self._zo_new_path = False

    def features(capture):
        """g_i up to the constant z_j: the feature column of the largest |z|."""
        reps = capture["reps"]
        return dict(zip(capture["ids"], reps[:, reps.abs().sum(0).argmax()].tolist(), strict=True))

    def select_microbatches(self, batch_samples):
        s = state(self)
        # Reference.
        original_flags(self)
        orig_select(self, [dict(b) for b in batch_samples])
        ref = self._check_capture
        ref_m = None if self.prev_m_t is None else self.prev_m_t.clone()
        ref_rng = torch.cuda.get_rng_state()
        restore(self, s)
        # Noise floor: same math, rows reversed.
        original_flags(self)
        orig_select(self, [_flip(b) for b in batch_samples])
        floor = self._check_capture
        restore(self, s)
        # Second noise floor: every row as its own micro-batch (keeps its padded width, so
        # the same divisor): the same math with different GEMM shapes for every row.
        original_flags(self)
        reps = [self.save_select(row) for b in batch_samples for row in _rows(b)]
        examples = [
            ex for b in batch_samples for ex in trainers._split_examples(trainers._to_cpu(b))
        ]
        budget = self._num_select_per_rank(len(batch_samples))
        self._select_across_ranks(torch.cat(reps).float().cpu(), examples, budget)
        rows = self._check_capture
        restore(self, s)
        # Configured path (continues).
        new_mbs = orig_select(self, batch_samples)
        new = self._check_capture
        entry = {"step": self.state.global_step}
        entry["selected"] = {
            "ref": ref["selected"],
            "floor": floor["selected"],
            "rows": rows["selected"],
            "new": new["selected"],
        }
        # Same set (floors: order of equal-gain picks may differ); the ordered check as well.
        entry["new_equals_ref"] = sorted(new["selected"]) == sorted(ref["selected"])
        entry["new_equals_ref_ordered"] = new["selected"] == ref["selected"]
        entry["floor_equals_ref"] = sorted(floor["selected"]) == sorted(ref["selected"])
        entry["rows_equals_ref"] = sorted(rows["selected"]) == sorted(ref["selected"])
        fr, ff, fn, fw = features(ref), features(floor), features(new), features(rows)
        computed = [i for i, g in fn.items() if g != 0.0]
        entry["zo_examples"] = len(computed)
        entry["num_examples"] = len(fn)

        def diffs(other):
            rel = [abs(other[i] - fr[i]) / max(abs(fr[i]), 1e-30) for i in computed]
            return {"max": max(rel, default=0.0), "median": float(np.median(rel)) if rel else 0.0}

        entry["g_rel_new"] = diffs(fn)
        entry["g_rel_floor"] = diffs(ff)
        entry["g_rel_rows"] = diffs(fw)
        if ref_m is not None and self.prev_m_t is not None:
            entry["adam_m_rel_new"] = _rel(self.prev_m_t, ref_m)
        entry["cuda_rng_equal"] = bool(torch.equal(torch.cuda.get_rng_state(), ref_rng))
        if self.state.global_step < grad_steps:
            entry.update(gradient_check(self, self._check_padded, new_mbs))
        record["steps"].append(entry)
        print(
            f"[check] {json.dumps({k: v for k, v in entry.items() if k != 'selected'})}", flush=True
        )
        return new_mbs

    orig_train_mbs = cls._train_microbatches

    def train_microbatches(self, selected_examples, num_batches):
        # Also keep the padded sub-batches of the same selection for the gradient check.
        saved = self.args.train_packing
        self.args.train_packing = "none"
        self._check_padded = orig_train_mbs(self, [dict(e) for e in selected_examples], num_batches)
        # One packed row per sub-batch: the fp32 eager arm (a merged row's T x T fp32
        # attention weights do not fit; merged == sub_batch math is in the CPU tests).
        self.args.train_packing = "sub_batch"
        self._check_sub = orig_train_mbs(self, [dict(e) for e in selected_examples], num_batches)
        self.args.train_packing = saved
        return orig_train_mbs(self, selected_examples, num_batches)

    cls._train_microbatches = train_microbatches

    def gradient_check(self, padded, packed):
        """Training gradients of the same selected examples, dropout off (eval mode).

        fp16-AMP arms: `ref` (padded, the model's attention; run twice for the GPU
        nondeterminism floor) and `new` (configured packing / attention). fp32 arms without
        autocast and with eager attention: `ref32` (padded) and `new32` (one packed row per
        sub-batch), which isolate the packing math from the fp16 kernels; `ref32` is the
        reference the fp16 arms are measured against (fp32 eager is ~1e-3 from fp64 on phi-2).
        All arms go through the same GradScaler, so their scales match.
        """
        model = self.model_wrapped if self.model_wrapped is not None else self.model
        params = [p for p in self.model.parameters() if p.requires_grad]
        rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        current_attn = self._attn_config._attn_implementation
        base_attn = self._check_model_attn

        def run(microbatches, attn, amp=True):
            self.model.zero_grad(set_to_none=True)
            self.model.eval()  # no dropout: comparable gradients
            saved_mode, saved_forward = self._train_mode, self.model.forward
            self._train_mode = lambda m: self._set_attention(attn)
            if not amp:
                self.model.forward = self.model._original_forward
            try:
                loss = sum(float(self._training_step(model, dict(mb))) for mb in microbatches if mb)
            finally:
                self._train_mode, self.model.forward = saved_mode, saved_forward
            grads = torch.cat([p.grad.double().flatten() for p in params if p.grad is not None])
            return loss, grads

        def cos(a, b):
            return float(a @ b / (a.norm() * b.norm()))

        loss_ref, g_ref = run(padded, base_attn)
        loss_rep, g_rep = run(padded, base_attn)
        loss_new, g_new = run(packed, self._train_attn or base_attn)
        loss_32, g_32 = run(padded, "eager", amp=False)
        loss_n32, g_n32 = run(self._check_sub, "eager", amp=False)
        self.model.zero_grad(set_to_none=True)
        self._set_attention(current_attn)
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])
        return {
            "loss_ref": loss_ref,
            "loss_rel_new": abs(loss_new - loss_ref) / abs(loss_ref),
            "loss_rel_repeat": abs(loss_rep - loss_ref) / abs(loss_ref),
            "loss_rel_new32_vs_ref32": abs(loss_n32 - loss_32) / abs(loss_32),
            "grad_rel_new": _rel(g_new, g_ref),
            "grad_rel_repeat": _rel(g_rep, g_ref),
            "grad_rel_new32_vs_ref32": _rel(g_n32, g_32),
            "grad_rel_ref_vs_ref32": _rel(g_ref, g_32),
            "grad_rel_new_vs_ref32": _rel(g_new, g_32),
            "grad_cos_ref_vs_ref32": cos(g_ref, g_32),
            "grad_cos_new_vs_ref32": cos(g_new, g_32),
            "grad_norm_ref": float(g_ref.norm()),
        }

    cls._select_microbatches = select_microbatches

    orig_init = cls.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self._check_model_attn = self._attn_config._attn_implementation
        record["attention"] = {
            "model": self._check_model_attn,
            "selection": self._zo_attn,
            "training": self._train_attn,
        }

    cls.__init__ = init
    import transformers

    transformers.Trainer.save_model = lambda self, *a, **k: None


def summarize(record: dict) -> dict:
    steps = record["steps"]
    grad = [s for s in steps if "grad_rel_new" in s]
    return {
        "steps": len(steps),
        "new_equals_ref": sum(s["new_equals_ref"] for s in steps),
        "floor_equals_ref": sum(s["floor_equals_ref"] for s in steps),
        "rows_equals_ref": sum(s["rows_equals_ref"] for s in steps),
        "g_rel_rows_max": max(s["g_rel_rows"]["max"] for s in steps),
        "g_rel_rows_median": float(np.median([s["g_rel_rows"]["median"] for s in steps])),
        "selected_overlap_rows": float(
            np.mean([len(set(s["selected"]["rows"]) & set(s["selected"]["ref"])) for s in steps])
        ),
        "g_rel_new_max": max(s["g_rel_new"]["max"] for s in steps),
        "g_rel_new_median": float(np.median([s["g_rel_new"]["median"] for s in steps])),
        "g_rel_floor_max": max(s["g_rel_floor"]["max"] for s in steps),
        "g_rel_floor_median": float(np.median([s["g_rel_floor"]["median"] for s in steps])),
        "zo_fraction": sum(s["zo_examples"] for s in steps) / sum(s["num_examples"] for s in steps),
        "cuda_rng_equal": all(s["cuda_rng_equal"] for s in steps),
        "grad_rel_new_max": max((s["grad_rel_new"] for s in grad), default=None),
        "grad_rel_repeat_max": max((s["grad_rel_repeat"] for s in grad), default=None),
        "grad_rel_new32_vs_ref32_max": max(
            (s["grad_rel_new32_vs_ref32"] for s in grad), default=None
        ),
        "grad_rel_ref_vs_ref32_mean": float(np.mean([s["grad_rel_ref_vs_ref32"] for s in grad]))
        if grad
        else None,
        "grad_rel_new_vs_ref32_mean": float(np.mean([s["grad_rel_new_vs_ref32"] for s in grad]))
        if grad
        else None,
        "grad_cos_ref_vs_ref32_mean": float(np.mean([s["grad_cos_ref_vs_ref32"] for s in grad]))
        if grad
        else None,
        "grad_cos_new_vs_ref32_mean": float(np.mean([s["grad_cos_new_vs_ref32"] for s in grad]))
        if grad
        else None,
        "selected_overlap_new": float(
            np.mean([len(set(s["selected"]["new"]) & set(s["selected"]["ref"])) for s in steps])
        ),
        "selected_overlap_floor": float(
            np.mean([len(set(s["selected"]["floor"]) & set(s["selected"]["ref"])) for s in steps])
        ),
        "selected_per_step": len(steps[0]["selected"]["ref"]),
        "loss_rel_new_max": max((s["loss_rel_new"] for s in grad), default=None),
        "loss_rel_repeat_max": max((s["loss_rel_repeat"] for s in grad), default=None),
    }


def main():
    config_path, out_path, *overrides = sys.argv[1:]
    with open(config_path) as f:
        config = json.load(f)
    grad_steps = 10**9
    for item in overrides:
        key, value = item.split("=", 1)
        if key == "grad_steps":
            grad_steps = int(value)
            continue
        config[key] = _parse(value)
    config["profile_timing"] = "off"
    record = {"config": config, "steps": []}
    install(record, grad_steps)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", dir=out_dir, prefix="check-config-", delete=False
    ) as f:
        json.dump(config, f, indent=1)
        merged = f.name
    from colm.train import train

    sys.argv = [sys.argv[0], merged]
    try:
        train.main()
    finally:
        os.remove(merged)
        record["summary"] = summarize(record) if record["steps"] else {}
        with open(out_path, "w") as f:
            json.dump(record, f, indent=1)
        print(f"[check] summary {json.dumps(record['summary'])}")
        print(f"check -> {out_path}")


if __name__ == "__main__":
    main()
