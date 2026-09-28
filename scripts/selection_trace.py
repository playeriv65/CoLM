#!/usr/bin/env python
"""Record per-step selection and training numbers of a CoLM run for A/B comparison.

    python -u scripts/selection_trace.py <config.json> <out.json> [key=value ...]

Runs `colm.train.train.main` in this process on the given config (plus JSON-typed
overrides, e.g. `max_steps=10 skip_unused_features=false`) with hooks that record,
per optimizer step:

* `selected`: dataset indices of the selected examples, in selection order;
* `feature_col`: the feature matrix column of the largest |z| coordinate, i.e. g_i * z_j
  for every gathered example (0 for examples whose feature was skipped);
* `train_losses`: the value returned by every `training_step` call;
* `grad_norm`: the pre-clip gradient norm.

The hooks only wrap methods that exist in every version of the trainer
(`_select_on_main`, `training_step`, `_clip_grad_norm`), so the same script traces the
reference implementation (run it with that checkout's environment) and the new one.
`save_model` is disabled so a trace writes nothing but the JSON (and the merged config
next to it). Compare two traces with `python scripts/selection_trace.py --compare a b`.
Two runs of the same code already diverge after a few steps (GPU nondeterminism, see F10
in docs/optimization-backlog.md); for exactness use scripts/check_exact_opts.py.
"""

import json
import os
import sys
import tempfile

import torch


def _parse(value: str):
    """JSON value (numbers, true/false/null, lists) or else the plain string."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _install_hooks(record: dict):
    import transformers

    import colm.train.trainers as trainers

    cls = trainers.SubsetTrainerEfficient
    orig_select, orig_step, orig_clip = (
        cls._select_on_main,
        cls.training_step,
        cls._clip_grad_norm,
    )

    def _current(self):
        steps = record["steps"]
        while len(steps) <= self.state.global_step:
            steps.append({"train_losses": []})
        return steps[self.state.global_step]

    def select_on_main(self, all_reps, complete_examples, total):
        idx, weights = orig_select(self, all_reps, complete_examples, total)
        entry = _current(self)
        entry["selected"] = [int(complete_examples[i]["indices"][0]) for i in idx]
        entry["gathered"] = [int(ex["indices"][0]) for ex in complete_examples]
        reps = all_reps.float()
        col = int(reps.abs().sum(dim=0).argmax())
        entry["feature_col_index"] = col
        entry["feature_col"] = reps[:, col].cpu().tolist()
        return idx, weights

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = orig_step(self, model, inputs, num_items_in_batch)
        _current(self)["train_losses"].append(float(loss))
        return loss

    def clip_grad_norm(self, model):
        grad_norm = orig_clip(self, model)
        _current(self)["grad_norm"] = float(grad_norm)
        return grad_norm

    cls._select_on_main = select_on_main
    cls.training_step = training_step
    cls._clip_grad_norm = clip_grad_norm
    transformers.Trainer.save_model = lambda self, *a, **k: None
    record["colm_file"] = trainers.__file__


def run(config_path: str, out_path: str, overrides: list[str]) -> None:
    with open(config_path) as f:
        config = json.load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        config[key] = _parse(value)
    config.setdefault("profile_timing", "off")
    record = {"config": config, "steps": []}
    _install_hooks(record)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", dir=out_dir, prefix="trace-config-", delete=False
    ) as f:
        json.dump(config, f, indent=1)
        merged = f.name
    from colm.train import train

    sys.argv = [sys.argv[0], merged]
    try:
        train.main()
    finally:
        os.remove(merged)
        with open(out_path, "w") as f:
            json.dump(record, f)
        print(f"trace -> {out_path} ({len(record['steps'])} steps, colm={record['colm_file']})")


def compare(path_a: str, path_b: str) -> None:
    a, b = (json.load(open(p))["steps"] for p in (path_a, path_b))
    print(f"{'step':>4} {'sel':>4} {'feat max rel':>12} {'loss max abs':>12} {'gnorm rel':>10}")
    for step, (sa, sb) in enumerate(zip(a, b, strict=False)):
        same = sa.get("selected") == sb.get("selected")
        fa, fb = torch.tensor(sa["feature_col"]), torch.tensor(sb["feature_col"])
        # Compare only the examples whose feature both runs computed.
        both = (fa != 0) & (fb != 0)
        rel = ((fa - fb).abs() / fa.abs().clamp_min(1e-30))[both]
        feat = float(rel.max()) if rel.numel() else 0.0
        la, lb = torch.tensor(sa["train_losses"]), torch.tensor(sb["train_losses"])
        loss = float((la.sum() - lb.sum()).abs())
        ga, gb = sa.get("grad_norm", float("nan")), sb.get("grad_norm", float("nan"))
        print(
            f"{step:>4} {'==' if same else 'DIFF':>4} {feat:>12.3e} {loss:>12.3e} "
            f"{abs(ga - gb) / abs(ga):>10.3e}"
        )
        if not same:
            print(f"     a: {sa.get('selected')}\n     b: {sb.get('selected')}")


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2], sys.argv[3:])
