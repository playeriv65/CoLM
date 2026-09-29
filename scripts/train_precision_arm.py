"""One short paired training run of the selection-precision study (`docs/selection-precision.md`).

The recipe is the rank sweep's r=128 / alpha=512 arm (`configs/rank_sweep/sweep.json`: its base
config, train overrides, environment and pinned flash kernel); only the steps, the evaluation
steps, the seed and the selection arm differ:

    F       fp32 selection prefix + fp32 suffix (library, selection_prefix_dtype=float32)
    P       fp16 prefix + fp32 suffix (library, selection_prefix_fp32_tail=0)
    P2      fp16 prefix with an fp32 tail of 2 blocks (the Phi-2 profile default)
    H       fp16 autocast for prefix and suffix (script-local patch of MezoEfficient.extract)
    random  uniform random 16 of the pool of 32 per step (script-local patch of the selector,
            the MeZO forward is skipped); same pools, same training compute
    random_kept  the selector's structure with a random ranking: kept sources always trained,
            the other picks random inside each source with the selector's quotas

    CUDA_VISIBLE_DEVICES=<gpu> python -u scripts/train_precision_arm.py --arm F --seed 0 \
        --steps 300 --eval-steps 100 200 300 --out-root $COLM_ARTIFACT_ROOT/artifacts/CoLM/precision-DATE
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SWEEP = "configs/rank_sweep/sweep.json"
# arm -> (selection_prefix_dtype, selection_prefix_fp32_tail); every arm pins both, whatever the
# sweep and the model profile default to.
ARMS = {
    "F": ("float32", 0),
    "P": ("float16", 0),
    "P2": ("float16", 2),
    "H": ("float16", 0),
    "random": ("float16", 0),
    "random_kept": ("float16", 0),
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[100, 200, 300])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=20260929)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=float, default=512)
    return parser.parse_args()


def main():
    args = arguments()
    spec = json.loads((REPO / SWEEP).read_text())
    os.environ.update(spec["env"])  # HF_HUB_OFFLINE etc. before transformers is imported
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from colm.jobs.rank_sweep import load_spec
    from colm.jobs.worker import local_kernel_env

    os.environ.update(local_kernel_env(spec))
    _, base = load_spec(SWEEP)
    name = f"{args.arm}-seed{args.seed}-{args.steps}steps"
    out = args.out_root / name
    out.mkdir(parents=True, exist_ok=True)
    config = {
        **base,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.steps,
        "eval_loss_steps": args.eval_steps,
        "seed": args.seed,
        "selection_prefix_dtype": ARMS[args.arm][0],
        "selection_prefix_fp32_tail": ARMS[args.arm][1],
        "output_dir": str(out),
        "run_name": name,
    }
    config_path = out / "train_config.json"
    config_path.write_text(json.dumps(config, indent=1) + "\n")

    import precision_arms

    if args.arm == "H":
        precision_arms.install_all_half()
    elif args.arm in ("random", "random_kept"):
        precision_arms.install_random_selection(args.random_seed, args.arm == "random_kept")
    print(f"ARM {args.arm}: {json.dumps(config)}", flush=True)

    from colm.train.train import main as train

    started, load = time.time(), os.getloadavg()
    train([str(config_path)])
    info = {
        "arm": args.arm,
        "seed": args.seed,
        "steps": args.steps,
        "random_seed": args.random_seed if args.arm.startswith("random") else None,
        "wall_s": time.time() - started,
        "loadavg_start": load,
        "loadavg_end": os.getloadavg(),
    }
    (out / "precision_arm.json").write_text(json.dumps(info, indent=1) + "\n")
    print("DONE " + json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
