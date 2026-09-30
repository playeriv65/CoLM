"""Correctness checks of the selection algorithms against independent references (CPU only).

Diagnostic for docs/paper-vs-code.md (part 1), not part of the library. Four checks:

1. Facility location. submodlib's `LazyGreedy` (as `colm.selection.facility_location` calls it)
   against a brute-force numpy greedy on the same similarity matrix (same picks, same order, same
   objective) and against the exact optimum by enumeration on small sets (greedy >= 1 - 1/e). The
   paper's objective `sum_i max_s [C - ||g_i - g_s||]` and the code's `max(dist) - dist`
   similarity pick the same set for every constant C (checked with C = 1e3 * max distance).
2. The similarity matrix: l1 / euclidean / cosine against a direct loop.
3. Per-source budgets: `class_budgets` against the upstream functions (`a6257b0`, read with
   `git show`, executed without their heavy imports) on random source compositions, including the
   compositions of real MathInstruct pools of 32 and 128 examples with the 10 kept sources set aside.
4. The Adam transform of `CoresetSelector._adam` against a float64 recursion written from the
   formulas of Sec. 4.2 (standard Adam: moments stored without bias correction, corrected when
   used) over a sequence of steps.

    python scripts/diagnostics/check_selection_algorithms.py [--out result.json]

Result: docs/paper-vs-code.md.
"""

import argparse
import ast
import itertools
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from submodlib import FacilityLocationFunction

from colm.selection.facility_location import class_budgets, get_orders_and_weights, similarity
from colm.selection.select import CoresetSelector
from colm.train.training_arguments import TrainingArguments

UPSTREAM = "a6257b0"
KEEP = [0, 1, 3, 5, 7, 8, 9, 10, 11, 13]


def upstream_functions(names_by_file: dict[str, list[str]], extra_globals: dict) -> dict:
    """Execute only the named top-level functions of upstream files (their imports are heavy)."""
    namespace = {"np": np, "torch": torch, **extra_globals}
    for path, names in names_by_file.items():
        text = subprocess.check_output(["git", "show", f"{UPSTREAM}:{path}"], text=True)
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module([node], []), path, "exec"), namespace)
    return namespace


def brute_greedy(S: np.ndarray, k: int):
    """Plain greedy on f(A) = sum_i max_{s in A} S[i, s]; returns (order, objective)."""
    n = len(S)
    chosen, best = [], np.full(n, -np.inf)
    for _ in range(k):
        gains = [
            (np.maximum(best, S[:, e]).sum() if e not in chosen else -np.inf) for e in range(n)
        ]
        e = int(np.argmax(gains))
        chosen.append(e)
        best = np.maximum(best, S[:, e])
    return chosen, float(best.sum())


def optimum(S: np.ndarray, k: int) -> float:
    return max(S[:, list(c)].max(axis=1).sum() for c in itertools.combinations(range(len(S)), k))


def first_divergence_is_tie(S, picked, order):
    """Whether the first differing pick of two greedy runs has an equal gain (float64)."""
    j = next(i for i, (a, b) in enumerate(zip(picked, order, strict=True)) if a != b)
    best = np.full(len(S), -np.inf)
    for e in order[:j]:
        best = np.maximum(best, S[:, e])
    gain = lambda e: np.maximum(best, S[:, e]).sum()  # noqa: E731
    return bool(abs(gain(picked[j]) - gain(order[j])) <= 1e-6 * abs(gain(order[j])))


def check_facility_location(rng):
    stats = {
        m: {"cases": 0, "same_order": 0, "tie_divergence": 0, "worst_rel_objective": 0.0}
        for m in ("l1", "euclidean", "cosine")
    }
    for _trial in range(100):
        n, d, k = int(rng.integers(20, 60)), int(rng.integers(2, 200)), int(rng.integers(2, 12))
        X = torch.from_numpy(rng.normal(size=(n, d)).astype(np.float32))
        for metric in stats:
            S = similarity(X, metric)
            flf = FacilityLocationFunction(
                n=n, sijs=S, separate_rep=False, mode="dense", metric=metric
            )
            picked = [
                g[0]
                for g in flf.maximize(
                    budget=k,
                    optimizer="LazyGreedy",
                    stopIfZeroGain=False,
                    stopIfNegativeGain=False,
                    show_progress=False,
                )
            ]
            S64 = S.astype(np.float64)
            order, objective = brute_greedy(S64, k)
            row = stats[metric]
            row["cases"] += 1
            row["same_order"] += picked == order
            if picked != order:
                row["tie_divergence"] += first_divergence_is_tie(S64, picked, order)
            row["worst_rel_objective"] = max(
                row["worst_rel_objective"],
                abs(S64[:, picked].max(axis=1).sum() - objective) / abs(objective),
            )
    ratios = []
    for _trial in range(20):
        n, k = 12, int(rng.integers(2, 5))
        X = torch.from_numpy(rng.normal(size=(n, 6)).astype(np.float32))
        S = similarity(X, "l1").astype(np.float64)
        _, greedy = brute_greedy(S, k)
        ratios.append(greedy / optimum(S, k))
    # the paper's C: any constant >= 0 leaves the greedy picks unchanged
    X = torch.from_numpy(rng.normal(size=(40, 30)).astype(np.float32))
    dist = torch.cdist(X, X, p=1)
    same_c = (
        brute_greedy((dist.max() - dist).double().numpy(), 8)[0]
        == brute_greedy((1e3 * dist.max() - dist).double().numpy(), 8)[0]
    )
    return {
        "lazygreedy_vs_brute_force": stats,
        "greedy_over_optimum_min": float(min(ratios)),
        "greedy_over_optimum_mean": float(np.mean(ratios)),
        "bound_1_minus_1_over_e": float(1 - 1 / np.e),
        "picks_independent_of_C": bool(same_c),
    }


def check_similarity(rng):
    X = rng.normal(size=(25, 17)).astype(np.float32)
    out = {}
    for metric in ("l1", "euclidean", "cosine"):
        S = similarity(torch.from_numpy(X), metric)
        D = np.array(
            [
                [np.abs(a - b).sum() if metric == "l1" else np.linalg.norm(a - b) for b in X]
                for a in X
            ]
        )
        if metric == "cosine":
            ref = np.array([[a @ b / np.linalg.norm(a) / np.linalg.norm(b) for b in X] for a in X])
        else:
            ref = D.max() - D
        out[metric] = float(np.abs(S - ref).max())
    return out


def real_source_pools(sizes, count, rng):
    """Source ids of random pools of MathInstruct (source frequencies of the data file)."""
    counts = np.array(
        [11239, 592, 89343, 1840, 28215, 7473, 49484, 300, 10632, 702, 9772, 14591, 24382, 13474]
    )
    probability = counts / counts.sum()
    return [rng.choice(14, size=size, p=probability) for size in sizes for _ in range(count)]


def check_budgets(rng):
    namespace = upstream_functions(
        {
            "colm/train/utils.py": [
                "convert_to_ordered_range",
                "increase_array_to_threshold",
                "increase_array_to_threshold_v2",
                "decrease_array_to_threshold",
            ]
        },
        {},
    )
    torchmetrics = types.ModuleType("torchmetrics")
    functional = types.ModuleType("torchmetrics.functional")
    functional.pairwise_cosine_similarity = lambda a, b: (
        torch.nn.functional.normalize(a, dim=1) @ torch.nn.functional.normalize(b, dim=1).T
    )
    torchmetrics.functional = functional
    sys.modules.setdefault("torchmetrics", torchmetrics)
    sys.modules.setdefault("torchmetrics.functional", functional)
    text = subprocess.check_output(
        ["git", "show", f"{UPSTREAM}:colm/train/facility_location.py"], text=True
    )
    text = "\n".join(line for line in text.splitlines() if "colm.train.utils" not in line)
    text = text.replace("from submodlib.submodlib import", "from submodlib import")
    upstream = dict(namespace)
    exec(compile(text, "upstream_facility_location", "exec"), upstream)

    cases, same_budget, same_pick, over_size, same_weights = 0, {"floor": 0, "ceil": 0}, 0, 0, 0
    per_start = {"floor": 0, "ceil": 0}
    pools = real_source_pools([32, 128], 40, rng)
    pools += [
        rng.integers(0, int(rng.integers(2, 8)), size=int(rng.integers(20, 130))) for _ in range(60)
    ]
    for y in pools:
        candidates = y[~np.isin(y, KEEP)] if len(y) in (32, 128) else y
        if len(candidates) < 4:
            continue
        budget = int(rng.integers(2, len(candidates)))  # B == n crashes submodlib (see below)
        for start in ("floor", "ceil"):
            try:
                _, _, mine = class_budgets(
                    budget, len(candidates), candidates, start, "proportional"
                )
            except ValueError:
                mine = None
            try:
                n = len(candidates)
                classes = np.unique(candidates)
                sizes = np.array([(candidates == c).sum() for c in classes])
                if start == "floor":
                    theirs = namespace["increase_array_to_threshold"](
                        np.int32(np.floor(sizes / n * budget)), budget
                    )
                else:
                    theirs = namespace["decrease_array_to_threshold"](
                        np.int32(np.ceil(sizes / n * budget)), budget
                    )
            except ValueError:
                theirs = None
            cases += 1
            per_start[start] += 1
            same_budget[start] += (mine is None and theirs is None) or (
                mine is not None and theirs is not None and np.array_equal(mine, theirs)
            )
            if mine is not None:
                over_size += int((mine > sizes).any())
        X = rng.normal(size=(len(candidates), 12)).astype(np.float32)
        order_m, weights_m = get_orders_and_weights(budget, torch.from_numpy(X), "l1", y=candidates)
        order_u, weights_u = upstream["get_orders_and_weights"](
            budget, torch.from_numpy(X), "l1", y=candidates
        )
        same_pick += np.array_equal(order_m, order_u)
        same_weights += np.array_equal(weights_m, weights_u)
    # float floor(size / n * B) against the exact integer floor(size * B / n), at the real budgets
    # (half of the pool minus the kept examples), and at B == n where the code crashes / miscounts
    wrong_floor, real_cases = 0, 0
    for y in real_source_pools([32, 128], 500, rng):
        candidates = y[~np.isin(y, KEEP)]
        budget = len(y) // 2 - int(np.isin(y, KEEP).sum())
        if budget <= 0 or len(candidates) == 0:
            continue
        sizes = np.bincount(np.unique(candidates, return_inverse=True)[1])
        real_cases += 1
        wrong_floor += int(
            (
                np.floor(sizes / len(candidates) * budget).astype(int)
                != (sizes * budget) // len(candidates)
            ).any()
        )
    at_full = 0
    for _ in range(2000):
        n = int(rng.integers(6, 130))
        y = rng.integers(0, int(rng.integers(2, 8)), size=n)
        _, _, counts = class_budgets(n, n, y, "floor", "proportional")
        at_full += int((counts > np.bincount(np.unique(y, return_inverse=True)[1])).any())
    return {
        "float_floor_differs_at_real_budgets": f"{wrong_floor} of {real_cases} pools",
        "budget_equals_pool_size_miscounts": f"{at_full} of 2000 compositions",
        "budget_cases": per_start,
        "budget_equal_to_upstream": same_budget,
        "budget_exceeds_source_size": over_size,
        "selection_cases": len(pools),
        "picks_equal_to_upstream": int(same_pick),
        "cluster_weights_equal_to_upstream": int(same_weights),
    }


def check_adam(rng):
    args = TrainingArguments(output_dir="unused")
    selector = CoresetSelector(args, num_layers=32)
    d, n, steps = 50, 12, 20
    b1, b2, eps = args.adam_beta1, args.adam_beta2, args.adam_epsilon
    prev_m = np.zeros(d)
    prev_v = np.zeros(d)
    worst = 0.0
    for t in range(steps):
        g = rng.normal(size=(n, d)) * rng.uniform(0.01, 1.0)
        feats = torch.from_numpy(g)
        out, m_t, v_t = selector._adam(feats, feats * feats, t)
        m = b1 * prev_m + (1 - b1) * g
        v = b2 * prev_v + (1 - b2) * g**2
        ref = (m / (1 - b1 ** (t + 1))) / (np.sqrt(v / (1 - b2 ** (t + 1))) + eps)
        worst = max(worst, float(np.abs(out.numpy() - ref).max()))
        chosen = rng.choice(n, size=n // 2, replace=False)
        selector.prev_m, selector.prev_v = m_t[chosen].mean(0), v_t[chosen].mean(0)
        prev_m, prev_v = m[chosen].mean(0), v[chosen].mean(0)
    # first step: the update is the sign of the gradient (m_hat / sqrt(v_hat) = g / |g|)
    g = rng.normal(size=(n, d))
    fresh = CoresetSelector(args, num_layers=32)
    first, _, _ = fresh._adam(torch.from_numpy(g), torch.from_numpy(g**2), 0)
    return {
        "max_abs_diff_over_steps": worst,
        "first_step_is_sign": float(np.abs(first.numpy() - np.sign(g)).max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    result = {
        "facility_location": check_facility_location(rng),
        "similarity_max_abs_diff": check_similarity(rng),
        "budgets_and_selection_vs_upstream": check_budgets(rng),
        "adam_transform": check_adam(rng),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
