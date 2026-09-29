"""Accuracy of g_i and the resulting selections, from the npz of `measure_selection_precision.py`.

CPU only. Prints markdown tables and writes `analysis.json` next to the input.

* g_i tables: every arm against the reference R (exact float64 directional derivative) pooled over
  directions x examples: Pearson, Spearman, sign agreement (count of examples whose sign differs),
  median and 90th percentile of |g - R| / |R|, and median |g - R| / median |R| (robust to R ~ 0).
* Selection: the library selector (default recipe: Adam transform, per-source coordinate mask,
  source-wise l1 facility location, keep sources) on the features g_i z of each arm, pools in
  order, its own Adam state carried across pools ("chain", as in training) and additionally with
  the state of R's chain given to every arm ("teacher-forced": only g_i differs). Overlap =
  |sel_X & sel_ref| per pool of 16, and over the facility-location picks alone (the kept
  sources are trained in full whatever g_i is and inflate the plain overlap).
"""

import argparse
import copy
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from colm.selection.facility_location import class_budgets
from colm.selection.select import CoresetSelector
from colm.train.training_arguments import TrainingArguments

SELECTION_ARMS = ["R", "F", "Fr", "P", "Pr", "H", "Hr", "F_e2", "P_e2", "H_e2"]
PAIRS = [("Fr", "F"), ("Pr", "P"), ("Hr", "H"), ("P", "F"), ("H", "F")]
SHOWN = ["F", "Fr", "P", "Pr", "H", "Hr", "F_e2", "P_e2", "H_e2", "Rfd3", "Rfd2"]
POOL = 32


def compare(x: np.ndarray, ref: np.ndarray) -> dict:
    """Agreement of an estimate `x` with `ref` (flattened over directions and examples)."""
    x, ref = x.ravel(), ref.ravel()
    rel = np.abs(x - ref) / np.abs(ref)
    third = np.argsort(np.abs(ref))
    small = third[: len(ref) // 3]
    return {
        "n": int(len(ref)),
        "pearson": float(np.corrcoef(x, ref)[0, 1]),
        "spearman": float(stats.spearmanr(x, ref)[0]),
        "sign_differs": int(np.sum(np.sign(x) != np.sign(ref))),
        "sign_differs_smallest_third": int(np.sum(np.sign(x[small]) != np.sign(ref[small]))),
        "rel_err_median": float(np.median(rel)),
        "rel_err_p90": float(np.percentile(rel, 90)),
        "abs_err_over_median_abs_ref": float(np.median(np.abs(x - ref)) / np.median(np.abs(ref))),
        "rel_l2": float(np.linalg.norm(x - ref) / np.linalg.norm(ref)),
    }


def chains(job):
    """Selections of every arm for one direction: chain, and teacher-forced on R's state."""
    direction, gs, z, sources, keep_ids = job
    torch.set_num_threads(4)
    args = TrainingArguments(output_dir="unused")
    z = torch.from_numpy(z).float()
    arms = list(gs)
    own = {arm: CoresetSelector(args, 32) for arm in arms}
    forced = {arm: CoresetSelector(args, 32) for arm in arms}
    out = {arm: {"chain": [], "forced": []} for arm in arms}
    pools = len(sources) // POOL
    for p in range(pools):
        window = slice(p * POOL, (p + 1) * POOL)
        src = sources[window].tolist()
        before = copy.copy(own["R"].state_dict())
        for arm in arms:
            feats = torch.from_numpy(gs[arm][window]).float().reshape(-1, 1) * z.reshape(1, -1)
            out[arm]["chain"].append(sorted(own[arm](feats, src, POOL // 2, p).indices))
            if arm == "R":
                out[arm]["forced"].append(out[arm]["chain"][-1])
                continue
            forced[arm].load_state_dict(before)
            out[arm]["forced"].append(sorted(forced[arm](feats, src, POOL // 2, p).indices))
    return direction, out


def overlaps(sel_a, sel_b, sources, keep_ids):
    """Per (direction, pool): plain overlap (of 16) and overlap of the facility-location picks."""
    plain, fl = [], []
    for p, (a, b) in enumerate(zip(sel_a, sel_b, strict=True)):
        src = sources[p * POOL : (p + 1) * POOL]
        free = {i for i in range(POOL) if int(src[i]) not in keep_ids}
        plain.append(len(set(a) & set(b)))
        fl.append((len(set(a) & set(b) & free), len(set(a) & free)))
    return plain, fl


def random_baselines(sources, keep_ids, args) -> dict:
    """Expected overlap of two independent random selections (a) 16 of 32, (b) random within the
    facility-location candidates with the same per-source quotas and the kept sources in full."""
    plain, fl_frac, fl_count = [], [], []
    for p in range(len(sources) // POOL):
        src = sources[p * POOL : (p + 1) * POOL]
        free = np.array([int(s) not in keep_ids for s in src])
        kept = int((~free).sum())
        budget = POOL // 2 - kept
        if budget <= 0:
            plain.append(POOL // 2)
            continue
        labels, classes, quotas = class_budgets(
            budget, int(free.sum()), src[free], args.num_per_class_start, "proportional"
        )
        sizes = np.bincount(labels, minlength=len(classes))
        expected = float(np.sum(quotas.astype(float) ** 2 / sizes))
        plain.append(kept + expected)
        fl_count.append(expected)
        fl_frac.append(expected / budget)
    return {
        "random_16_of_32_overlap_of_16": (POOL // 2) ** 2 / POOL,
        "random_within_candidates_overlap_of_16": float(np.mean(plain)),
        "random_within_candidates_fl_fraction": float(np.mean(fl_frac)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", type=Path)
    parser.add_argument("--no-selection", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--arms", help="comma-separated arms to show and select (default: Phase 1)")
    parser.add_argument("--pairs", help="comma-separated A:B pairs (A against reference B)")
    parser.add_argument("--out", type=Path, help="analysis json (default: next to the npz)")
    opts = parser.parse_args()
    shown = opts.arms.split(",") if opts.arms else SHOWN
    selected = ["R"] + [a for a in shown if a != "R"] if opts.arms else SELECTION_ARMS
    pairs = [tuple(p.split(":")) for p in opts.pairs.split(",")] if opts.pairs else PAIRS
    data = np.load(opts.npz)
    g = {k[2:]: data[k] for k in data.files if k.startswith("g_")}
    done = int(data["pools_done"]) * POOL
    g = {k: v[:, :done] for k, v in g.items()}
    sources = data["source"][:done]
    args = TrainingArguments(output_dir="unused")
    keep_ids = set(args.keep_source_ids)
    result = {"directions": int(g["R"].shape[0]), "examples": done, "g": {}, "pairs": {}}

    reference = g["R"]
    print(
        f"R: {reference.shape[0]} directions x {done} examples; median |g| "
        f"{np.median(np.abs(reference)):.3e}\n"
    )
    print(
        "| arm | Pearson | Spearman | sign differs | (smallest third) | median rel err | "
        "p90 rel err | median abs err / median abs g | rel L2 |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in shown:
        if arm not in g:
            continue
        m = result["g"][arm] = compare(g[arm], reference)
        print(
            f"| {arm} | {m['pearson']:.4f} | {m['spearman']:.4f} | {m['sign_differs']}/{m['n']} "
            f"| {m['sign_differs_smallest_third']} | {m['rel_err_median']:.3g} | "
            f"{m['rel_err_p90']:.3g} | {m['abs_err_over_median_abs_ref']:.3g} | "
            f"{m['rel_l2']:.3g} |"
        )
    print("\nPairs (second is the reference):\n")
    for a, b in pairs:
        if a in g and b in g:
            m = result["pairs"][f"{a}_vs_{b}"] = compare(g[a], g[b])
            print(
                f"- {a} vs {b}: Pearson {m['pearson']:.4f}, sign differs {m['sign_differs']}/"
                f"{m['n']}, median rel err {m['rel_err_median']:.3g}, rel L2 {m['rel_l2']:.3g}"
            )

    if not opts.no_selection:
        arms = [a for a in selected if a in g]
        jobs = [
            (d, {a: g[a][d] for a in arms}, data["z"][d], sources, keep_ids)
            for d in range(reference.shape[0])
        ]
        with ProcessPoolExecutor(opts.workers) as pool:
            selections = dict(pool.map(chains, jobs))
        result["random"] = random_baselines(sources, keep_ids, args)
        print("\nRandom baselines:", json.dumps(result["random"]))
        n_pools = done // POOL
        table = {}
        comparisons = [(a, "R") for a in arms if a != "R"] + [
            pair for pair in pairs if pair[0] in g and pair[1] in g
        ]
        print(
            "\n| arm vs ref | chain overlap /16 | chain FL overlap | forced overlap /16 | "
            "forced FL overlap |"
        )
        print("|---|---|---|---|---|")
        for a, b in comparisons:
            row = {}
            for mode in ("chain", "forced"):
                plain, fl = [], []
                for d in selections:
                    pl, f = overlaps(
                        selections[d][a][mode], selections[d][b][mode], sources, keep_ids
                    )
                    plain += pl
                    fl += f
                hit, total = np.sum([x[0] for x in fl]), np.sum([x[1] for x in fl])
                row[mode] = {
                    "overlap_of_16": float(np.mean(plain)),
                    "fl_overlap_fraction": float(hit / total),
                    "pools": len(plain),
                }
            table[f"{a}_vs_{b}"] = row
            print(
                f"| {a} vs {b} | {row['chain']['overlap_of_16']:.2f} | "
                f"{row['chain']['fl_overlap_fraction']:.3f} | "
                f"{row['forced']['overlap_of_16']:.2f} | "
                f"{row['forced']['fl_overlap_fraction']:.3f} |"
            )
        result["selection"] = table
        result["selection_pools_per_direction"] = n_pools
        result["selections"] = {
            str(d): {a: selections[d][a] for a in selections[d]} for d in selections
        }
    out = opts.out or opts.npz.with_name("analysis.json")
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
