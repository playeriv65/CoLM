"""Cost of the last-layer feature options, on one pool (rough: a shared GPU, one run, no repeats).

Diagnostic for docs/paper-vs-code.md (what a change of the selection feature would cost), not part
of the library. On the first `--pool` examples of the pool file (packs of `--pack-tokens`) it times,
with CUDA events, from the same fp32 prefix state (computed once per pack):

* `prefix`   the decoder up to the last layer (fp16 autocast, fp32 tail 2: the library default);
* `fd_m`     m directional derivatives of the library (two last-layer replays per direction);
* `exact`    the exact per-example gradient w.r.t. the MeZO parameter (one forward through last
             layer + head, one backward per example; MATH attention).

Timing protocol of the audit (`docs/optimization-backlog.md`) is not applied: the numbers are a
first estimate of ratios on a loaded card, not a benchmark.

    python -u scripts/diagnostics/time_last_layer_options.py --config configs/diagnostics/prefix_precision_phi2.json \
        --pool-file $ROOT/inputs/pools.pkl --adapter $ROOT/inputs/adapter_model.safetensors --device cuda:0

Result: docs/paper-vs-code.md.
"""

import argparse
import json
import pickle
from pathlib import Path

import torch
from measure_selection_precision import EPS, load_model, pool_examples
from measure_true_gradients import prefix_fp32, true_gradients
from precision_arms import fd_estimate, to_device
from transformers import AutoTokenizer

from colm.selection.packing import greedy_groups, pack
from colm.selection.zo import LastLayerSplit, Perturbation, zo_parameters


def timed(fn, repeats=3):
    fn()  # warm-up
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pool-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--pack-tokens", type=int, default=1536)
    parser.add_argument("--pool", type=int, default=0)
    parser.add_argument("--directions", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    recipe = json.loads(args.config.read_text())
    tokenizer = AutoTokenizer.from_pretrained(recipe["model_name_or_path"], local_files_only=True)
    pools = pickle.load(args.pool_file.open("rb"))
    model = load_model(recipe, args.adapter, device)
    split = LastLayerSplit(model.get_base_model())
    (name, param), *_ = zo_parameters(model, ["v_proj"], -1)
    rel = split.relative_name(name)
    examples = pool_examples(tokenizer, pools, args.pool)
    groups = greedy_groups([len(e) for e in examples], args.pack_tokens)
    batches = [to_device(pack([examples[i] for i in g]), device) for g in groups]
    states = [prefix_fp32(split, b, 2) for b in batches]
    zs = Perturbation([(name, param)], EPS, 1).z()
    rows = {
        "packs": len(batches),
        "tokens": sum(len(e) for e in examples),
        "examples": len(examples),
    }
    rows["prefix_ms"] = sum(timed(lambda b=b: prefix_fp32(split, b, 2)) for b in batches)
    for m in args.directions:
        directions = [[torch.randn_like(z) for z in zs] if k else zs for k in range(m)]

        def run(m=m, directions=directions):
            for b, s in zip(batches, states, strict=True):
                for z in directions:
                    fd_estimate(split, s, b, [name], [param.detach()], z, EPS)

        rows[f"fd_m{m}_ms"] = timed(run, repeats=1 if m > 16 else 3)
    rows["exact_ms"] = sum(
        timed(lambda b=b, s=s: true_gradients(split, s, b, rel, param.detach()), repeats=2)
        for b, s in zip(batches, states, strict=True)
    )
    print(json.dumps(rows, indent=1), flush=True)


if __name__ == "__main__":
    main()
