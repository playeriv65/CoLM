"""Does the single-direction MeZO feature carry the geometry of the true gradients? (T1 / T2 / T3, CPU)

Analysis of the gradients saved by `measure_true_gradients.py` for docs/paper-vs-code.md. Nothing
here is library code; the selection runs through the library's own `CoresetSelector` (Adam
transform, per-source coordinate mask, source-wise facility location), so a variant differs from
the recipe only in the features it is given.

Parts (`--parts`):

  geometry  T1 a/c, all pairs of the 16 pools of 32: the true pairwise geometry (l2, cosine, l1),
            the fraction of a gradient the direction z keeps (|z.g| / (|z||g|), expected 0.8 / sqrt(d)),
            and how well the distances of m random projections (m = 1, 4, 16, 64, 256; the
            recipe's fixed z0 and fresh draws) reproduce the true l2 distances (Pearson / Spearman).
  steps     T1 b/d and T2, sequences of pools of 128 (the 4-GPU selection pool: 4 pools of the pool
            file, then the sampled pools): every variant is a chain of `CoresetSelector`s with its own
            Adam moments, selecting 64 of 128 per step. Variants: `oracle` (true gradients), oracle
            design variants, `z0` (the code: fixed z, exact directional derivative), `z0_fd` /
            `z0_fd_rev` (the code's finite-difference estimate, pool packed in order / reversed:
            its repeat noise), `fresh` / `fresh_b` (new z every step, two independent draws),
            `fresh_sgd`, `m4` / `m16` / `m64` / `m256` (m shared directions per step, features = the m
            directional derivatives), `m16_fixed`, `perex` (a different z per example: a control
            whose features are not comparable). Reported: overlap of the picks with the oracle's
            against random-within-candidates, agreement between variants, correlation of the
            facility-location distances with the oracle's, the rank-1 structure of the features, what
            the coordinate mask selects, and the gradient-matching error of every selected set
            (|mean g over the 64 - mean g over the 128| / |mean g over the 128|; random 64 as
            reference).
  zdraw     T2: the fixed-z chain repeated with many independent z (each fixed over the sequence):
            is the recipe's z0 a typical draw, and how much does the agreement with the oracle
            depend on the draw.
  mean      T3: the sample mean of m directional estimates (g.z_j) z_j against the true gradient:
            cosine at m = 1, 16, 256, 4096, and the theoretical sqrt(m / (m + d)).
  weights   loss normalisation: the step's gradient with the global token mean of the library against
            the upstream mean of per-micro-batch token means.

    python -u scripts/diagnostics/analyze_z_geometry.py --root $COLM_ARTIFACT_ROOT/artifacts/CoLM/paper-review-DATE \
        --parts geometry steps zdraw mean weights --out $ROOT/analysis.json

Result: docs/paper-vs-code.md.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

import colm.selection.select as select_module
from colm.selection.facility_location import class_budgets
from colm.selection.select import CoresetSelector
from colm.train.training_arguments import TrainingArguments

torch.set_num_threads(16)
POOL = 128
SELECT = 64


# ----- data ---------------------------------------------------------------------------------
class Pool:
    """One selection pool: true gradients [n, d] (float32 torch) and the per-example records."""

    def __init__(self, gradients, meta, rows):
        self.G = torch.from_numpy(np.asarray(gradients[rows], dtype=np.float32))
        self.rows = rows
        self.sources = meta["source"][rows].astype(int).tolist()
        self.s_exact = torch.from_numpy(meta["s_exact"][rows])
        self.g_code = torch.from_numpy(meta["g_code"][rows])
        self.g_code_rev = torch.from_numpy(meta["g_code_rev"][rows])
        self.loss = meta["loss"][rows]
        self.labels = meta["num_labels"][rows]

    def __len__(self):
        return len(self.sources)


def load(root: Path):
    z0 = torch.from_numpy(np.load(root / "z0.npy"))
    pools32, pools128 = [], []
    G = np.load(root / "G_pools.npy", mmap_mode="r")
    meta = dict(np.load(root / "meta_pools.npz"))
    pools32 = [Pool(G, meta, np.arange(i * 32, (i + 1) * 32)) for i in range(len(G) // 32)]
    pools128 = [Pool(G, meta, np.arange(i * POOL, (i + 1) * POOL)) for i in range(len(G) // POOL)]
    if (root / "meta_extra.npz").exists():
        Ge = np.load(root / "G_extra.npy", mmap_mode="r")
        me = dict(np.load(root / "meta_extra.npz"))
        done = int(np.isfinite(me["gnorm"]).sum()) // POOL
        pools128 += [Pool(Ge, me, np.arange(i * POOL, (i + 1) * POOL)) for i in range(done)]
    return z0, pools32, pools128


def selector_args(**changes):
    args = TrainingArguments(output_dir="unused")
    args = copy.copy(args)
    for key, value in changes.items():
        setattr(args, key, value)
    return args


def draw(rng, *shape):
    return torch.from_numpy(rng.standard_normal(shape).astype(np.float32))


# ----- statistics helpers -------------------------------------------------------------------
def pair_vector(D: torch.Tensor) -> np.ndarray:
    n = len(D)
    return D[torch.triu_indices(n, n, 1).unbind()].double().numpy()


def correlations(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(a, b)[0, 1]), float(spearmanr(a, b)[0])


def summarise(values) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if not len(v):
        return {"n": 0}
    return {
        "mean": float(v.mean()),
        "std": float(v.std()),
        "p05": float(np.percentile(v, 5)),
        "p50": float(np.percentile(v, 50)),
        "p95": float(np.percentile(v, 95)),
        "n": int(len(v)),
    }


def random_overlap(sources_candidates, quotas_by_source) -> float:
    """Expected |A n B| of two independent random selections with the per-source quotas."""
    counts = {}
    for s in sources_candidates:
        counts[s] = counts.get(s, 0) + 1
    return sum(k * k / counts[s] for s, k in quotas_by_source.items() if counts.get(s))


# ----- geometry of the pools of 32 -----------------------------------------------------------
def geometry(pools32, z0, rng, draws_fresh=200, draws_multi=12):
    """Statistics of the true gradients of every pool of 32, and the projection-distance fidelity."""
    d = z0.numel()
    keys = ("cv_norm", "mean_cos", "cos_to_mean", "top1", "top3", "top10", "pr_centered", "ret_z0")
    rows = {k: [] for k in keys}
    truth_agree = {"l2_vs_cos": [], "l2_vs_l1": []}
    true_l2, norms_all = [], []
    for pool in pools32:
        G = pool.G
        norms = G.norm(dim=1)
        norms_all.append(norms)
        rows["cv_norm"].append(float(norms.std() / norms.mean()))
        Gn = G / norms[:, None]
        cos = Gn @ Gn.T
        n = len(G)
        rows["mean_cos"].append(float((cos.sum() - n) / (n * (n - 1))))
        mean = G.mean(0)
        rows["cos_to_mean"].append(float(((G @ mean) / norms / mean.norm()).mean()))
        sv = torch.linalg.svdvals(G.double()) ** 2
        rows["top1"].append(float(sv[0] / sv.sum()))
        rows["top3"].append(float(sv[:3].sum() / sv.sum()))
        rows["top10"].append(float(sv[:10].sum() / sv.sum()))
        svc = torch.linalg.svdvals((G - mean).double()) ** 2
        rows["pr_centered"].append(float(svc.sum() ** 2 / (svc**2).sum()))
        rows["ret_z0"].append(float(((G @ z0).abs() / norms / z0.norm()).mean()))
        D2 = torch.cdist(G, G)
        truth_agree["l2_vs_cos"].append(correlations(pair_vector(D2), pair_vector(1 - cos))[1])
        truth_agree["l2_vs_l1"].append(
            correlations(pair_vector(D2), pair_vector(torch.cdist(G, G, p=1)))[1]
        )
        true_l2.append(pair_vector(D2))

    proj = {}

    def add(key, pool_index, S):
        proj.setdefault(key, []).append(
            correlations(pair_vector(torch.cdist(S, S)), true_l2[pool_index])
        )

    for i, pool in enumerate(pools32):  # the recipe's z0
        add((1, "z0"), i, (pool.G @ z0)[:, None])
    retention = []
    for _ in range(draws_fresh):  # fresh z, the same draw for every pool
        z = draw(rng, d)
        for i, pool in enumerate(pools32):
            s = pool.G @ z
            add((1, "fresh"), i, s[:, None])
            retention.append(float((s.abs() / norms_all[i] / z.norm()).mean()))
    for m in (4, 16, 64, 256):
        for _ in range(draws_multi if m < 256 else 4):
            Z = draw(rng, d, m)
            for i, pool in enumerate(pools32):
                add((m, "fresh"), i, pool.G @ Z / np.sqrt(m))
    for i, pool in enumerate(pools32):  # control: a different z for every example
        S = torch.stack([pool.G[j] @ draw(rng, d) for j in range(len(pool))])
        add((1, "z_per_example"), i, S[:, None])
    result = {k: summarise(v) for k, v in rows.items()}
    result["ret_fresh"] = summarise(retention)
    result["truth_spearman_l2_vs"] = {k: summarise(v) for k, v in truth_agree.items()}
    result["projection_distance_vs_true_l2"] = {
        f"m={m},{kind}": {
            "pearson": summarise([p for p, _ in v]),
            "spearman": summarise([s for _, s in v]),
        }
        for (m, kind), v in sorted(proj.items())
    }
    result["d"] = d
    result["expected_retention_1z"] = float(np.sqrt(2 / np.pi) / np.sqrt(d))
    return result


# ----- chains of selectors ---------------------------------------------------------------------
CAPTURE = {}
_ORIGINAL_FL = select_module.get_orders_and_weights
_ORIGINAL_RANK = CoresetSelector._rank


def _capture_fl(B, X, metric, y=None, per_class_start="floor", strategy="proportional"):
    CAPTURE["X"], CAPTURE["y"] = X.detach().cpu(), None if y is None else np.array(y)
    return _ORIGINAL_FL(B, X, metric, y=y, per_class_start=per_class_start, strategy=strategy)


def _capture_rank(self, importance):
    out = _ORIGINAL_RANK(self, importance)
    CAPTURE.setdefault("ranks", []).append(
        out.cpu() if torch.is_tensor(out) else torch.as_tensor(out)
    )
    return out


select_module.get_orders_and_weights = _capture_fl
CoresetSelector._rank = _capture_rank


class Chain:
    """A CoresetSelector with its own Adam moments and a feature function."""

    def __init__(self, name, features, **changes):
        self.name, self.features = name, features
        self.args = selector_args(data_selection_method="weightedsubmodlib", **changes)
        self.selector = CoresetSelector(self.args, 32)
        self.picks, self.X, self.y, self.ranks = [], None, None, []

    def step(self, t, pool, ctx):
        feats = self.features(t, pool, ctx)
        CAPTURE.clear()
        selection = self.selector(feats, pool.sources, SELECT, t)
        n_keep = int(np.isin(pool.sources, self.args.keep_source_ids).sum())
        self.picks = selection.indices[n_keep:]
        self.all = selection.indices
        self.weights = selection.weights
        self.X, self.y = CAPTURE["X"], CAPTURE["y"]
        self.ranks = CAPTURE.get("ranks", [])
        return selection


def within_source_l1(X, y):
    """Vector of the l1 distances between the examples of the same source (candidates order)."""
    parts = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) > 1:
            parts.append(pair_vector(torch.cdist(X[idx].float(), X[idx].float(), p=1)))
    return np.concatenate(parts) if parts else np.zeros(0)


def within_source_distance(G, sources, keep, kind="l2"):
    """Distances of the true gradients within the source, candidate examples only, same order."""
    cand = np.array([i for i, s in enumerate(sources) if s not in keep])
    y = np.array([sources[i] for i in cand])
    parts = []
    for c in sorted(np.unique(y)):
        idx = cand[y == c]
        if len(idx) > 1:
            g = G[idx]
            parts.append(pair_vector(torch.cdist(g, g)))
    return np.concatenate(parts) if parts else np.zeros(0)


def gradient_error(G, chosen, pool_mean):
    mean = G[chosen].mean(0)
    err = float((mean - pool_mean).norm() / pool_mean.norm())
    cos = float(mean @ pool_mean / mean.norm() / pool_mean.norm())
    return err, cos


def weighted_error(G, indices, weights, pool_mean):
    """Error of the weighted mean of the selected gradients (cluster sizes as weights)."""
    w = torch.tensor(weights, dtype=torch.float32)
    mean = (w[:, None] * G[indices]).sum(0) / w.sum()
    return float((mean - pool_mean).norm() / pool_mean.norm())


def rank1_energy(X, y, centered=False):
    """Mean over the sources of the energy share of the first singular value of their features
    (`centered`: after removing the mean feature of the source, i.e. the variation between examples)."""
    shares = []
    for c in np.unique(y):
        block = X[np.where(y == c)[0]].double()
        if centered:
            block = block - block.mean(0)
        if len(block) >= 3:
            sv = torch.linalg.svdvals(block) ** 2
            shares.append(float(sv[0] / sv.sum()))
    return float(np.mean(shares))


def by_key(pool, cand, labels, classes, quotas, key, largest):
    """The `quota` examples with the smallest / largest `key` of every source (candidates only)."""
    picks = []
    for c, k in zip(classes, quotas, strict=True):
        members = [cand[j] for j in np.where(labels == c)[0]]
        order = sorted(members, key=lambda i: key[i], reverse=largest)
        picks += order[: int(k)]
    return picks


def build_chains(z0, rng, d):
    def oracle(t, p, c):
        return p.G

    def z0_exact(t, p, c):
        return torch.outer(p.s_exact.float(), z0)

    def z0_fd(t, p, c):
        return torch.outer(p.g_code.float(), z0)

    def z0_fd_rev(t, p, c):
        return torch.outer(p.g_code_rev.float(), z0)

    def make_fresh(seed):
        gen = np.random.default_rng(seed)

        def f(t, p, c):
            z = draw(gen, d)
            return torch.outer(p.G @ z, z)

        return f

    def make_multi(m, seed, fixed=False):
        gen = np.random.default_rng(seed)
        Z0 = draw(gen, d, m)

        def f(t, p, c):
            Z = Z0 if fixed else draw(gen, d, m)
            return p.G @ Z

        return f

    def perex(t, p, c):
        gen = np.random.default_rng(1000 + t)
        out = torch.empty(len(p), d)
        for i in range(len(p)):
            z = draw(gen, d)
            out[i] = (p.G[i] @ z) * z
        return out

    seed = int(rng.integers(1 << 30))
    return [
        Chain("oracle", oracle),
        Chain("oracle_sgd", oracle, mezo_optim="sgd"),
        Chain("oracle_full_cos", oracle, mezo_optim="sgd", zo_dim=d, facility_similarity="cosine"),
        Chain(
            "oracle_full_l2", oracle, mezo_optim="sgd", zo_dim=d, facility_similarity="euclidean"
        ),
        Chain("z0", z0_exact),
        Chain("z0_fd", z0_fd),
        Chain("z0_fd_rev", z0_fd_rev),
        Chain("fresh", make_fresh(seed + 1)),
        Chain("fresh_b", make_fresh(seed + 2)),
        Chain("fresh_sgd", make_fresh(seed + 3), mezo_optim="sgd"),
        Chain("m4", make_multi(4, seed + 4), zo_dim=4),
        Chain("m16", make_multi(16, seed + 5), zo_dim=16),
        Chain("m64", make_multi(64, seed + 6), zo_dim=64),
        Chain("m256", make_multi(256, seed + 8), zo_dim=256),
        Chain("m16_fixed", make_multi(16, seed + 7, fixed=True), zo_dim=16),
        Chain("perex", perex),
    ]


def steps(pools128, z0, rng):
    d = z0.numel()
    chains = build_chains(z0, rng, d)
    keep = chains[0].args.keep_source_ids
    names = [c.name for c in chains]
    overlap = {n: [] for n in names}
    match = {
        n: [] for n in names + ["random_keep_aware", "random_plain", "oracle_full_pool_kept_only"]
    }
    match_cos = {n: [] for n in match}
    match_w = {n: [] for n in names + ["random_stratified"]}
    pseudo = ["norm_low", "norm_high", "loss_low", "loss_high"]
    pseudo_overlap = {f"{p}~{r}": [] for p in pseudo for r in ("oracle", "z0")}
    for p in pseudo:
        match[p], match_cos[p] = [], []
    corr = {
        k: [] for k in ("abs_s_vs_gnorm", "s_vs_mean_projection", "abs_s_vs_loss", "gnorm_vs_loss")
    }
    geo = {n: {"pearson": [], "spearman": [], "vs_true_l2_spearman": []} for n in names}
    rank1 = {n: [] for n in names}
    rank1c = {n: [] for n in names}
    overlap_l2 = {n: [] for n in names}
    mask_z = []
    pair_agree = {}
    random_expect, picks_count = [], []
    t0 = time.time()
    for t, pool in enumerate(pools128):
        sources = pool.sources
        pool_mean = pool.G.mean(0)
        cand = [i for i, s in enumerate(sources) if s not in keep]
        n_keep = len(sources) - len(cand)
        labels, classes, quotas = class_budgets(
            SELECT - n_keep,
            len(cand),
            np.array([sources[i] for i in cand]),
            "floor",
            "proportional",
        )
        quota_by_source = {
            int(np.unique([sources[i] for i in cand])[c]): int(k)
            for c, k in zip(classes, quotas, strict=True)
        }
        random_expect.append(random_overlap([sources[i] for i in cand], quota_by_source))
        picks_count.append(SELECT - n_keep)
        for chain in chains:
            chain.step(t, pool, {})
        oracle = chains[0]
        full_l2 = next(c for c in chains if c.name == "oracle_full_l2")
        true_l2 = within_source_distance(pool.G, sources, keep)
        ref = within_source_l1(oracle.X, oracle.y)
        for chain in chains:
            overlap[chain.name].append(len(set(chain.picks) & set(oracle.picks)))
            err, cos = gradient_error(pool.G, chain.all, pool_mean)
            match[chain.name].append(err)
            match_cos[chain.name].append(cos)
            if chain.X.shape[1] >= 1 and chain.name not in ("oracle_full_cos", "oracle_full_l2"):
                mine = within_source_l1(chain.X, chain.y)
                if len(mine) == len(ref):
                    p, s = correlations(mine, ref)
                    geo[chain.name]["pearson"].append(p)
                    geo[chain.name]["spearman"].append(s)
                    geo[chain.name]["vs_true_l2_spearman"].append(correlations(mine, true_l2)[1])
            rank1[chain.name].append(rank1_energy(chain.X, chain.y))
            rank1c[chain.name].append(rank1_energy(chain.X, chain.y, centered=True))
            overlap_l2[chain.name].append(len(set(chain.picks) & set(full_l2.picks)))
            match_w[chain.name].append(weighted_error(pool.G, chain.all, chain.weights, pool_mean))
        for a, b in (
            ("z0", "fresh"),
            ("fresh", "fresh_b"),
            ("z0", "z0_fd"),
            ("z0_fd", "z0_fd_rev"),
            ("z0", "z0_fd_rev"),
            ("oracle", "oracle_sgd"),
            ("oracle", "oracle_full_cos"),
            ("oracle_full_cos", "oracle_full_l2"),
            ("z0", "m16"),
            ("z0", "m16_fixed"),
            ("fresh", "m16"),
        ):
            ca = next(c for c in chains if c.name == a)
            cb = next(c for c in chains if c.name == b)
            pair_agree.setdefault(f"{a}~{b}", []).append(len(set(ca.picks) & set(cb.picks)))
        ranks = next(c for c in chains if c.name == "z0").ranks
        if ranks:
            sel = torch.unique(torch.cat([r.flatten() for r in ranks]))
            mask_z.append(float(z0[sel].abs().mean()))
        gnorm = pool.G.norm(dim=1).numpy()
        keyed = {
            "norm_low": (gnorm, False),
            "norm_high": (gnorm, True),
            "loss_low": (pool.loss, False),
            "loss_high": (pool.loss, True),
        }
        z0_chain = next(c for c in chains if c.name == "z0")
        for name, (key, largest) in keyed.items():
            picks = by_key(pool, cand, labels, classes, quotas, key, largest)
            pseudo_overlap[f"{name}~oracle"].append(len(set(picks) & set(oracle.picks)))
            pseudo_overlap[f"{name}~z0"].append(len(set(picks) & set(z0_chain.picks)))
            err, cos = gradient_error(
                pool.G, [i for i in range(len(sources)) if sources[i] in keep] + picks, pool_mean
            )
            match[name].append(err)
            match_cos[name].append(cos)
        cand_idx = np.array(cand)
        s_ex = pool.s_exact.numpy()
        proj = (pool.G @ pool_mean).numpy()
        corr["abs_s_vs_gnorm"].append(float(spearmanr(np.abs(s_ex[cand_idx]), gnorm[cand_idx])[0]))
        corr["s_vs_mean_projection"].append(float(spearmanr(s_ex[cand_idx], proj[cand_idx])[0]))
        corr["abs_s_vs_loss"].append(
            float(spearmanr(np.abs(s_ex[cand_idx]), pool.loss[cand_idx])[0])
        )
        corr["gnorm_vs_loss"].append(float(spearmanr(gnorm[cand_idx], pool.loss[cand_idx])[0]))
        # random references: 64 of 128 uniformly, and the keep-aware random selection
        errs_plain, errs_aware = [], []
        for _ in range(100):
            chosen = rng.choice(len(sources), size=SELECT, replace=False).tolist()
            errs_plain.append(gradient_error(pool.G, chosen, pool_mean))
            chosen = [i for i, s in enumerate(sources) if s in keep]
            for c, k in zip(classes, quotas, strict=True):
                members = [cand[j] for j in np.where(labels == c)[0]]
                chosen += rng.choice(members, size=int(k), replace=False).tolist()
            errs_aware.append(gradient_error(pool.G, chosen, pool_mean))
        strat = []
        for _ in range(100):
            chosen = [i for i, s in enumerate(sources) if s in keep]
            weights = [1.0] * len(chosen)
            for c, k in zip(classes, quotas, strict=True):
                members = [cand[j] for j in np.where(labels == c)[0]]
                chosen += rng.choice(members, size=int(k), replace=False).tolist()
                weights += [len(members) / int(k)] * int(k)
            strat.append(weighted_error(pool.G, chosen, weights, pool_mean))
        match_w["random_stratified"].append(float(np.mean(strat)))
        match["random_plain"].append(float(np.mean([e for e, _ in errs_plain])))
        match_cos["random_plain"].append(float(np.mean([c for _, c in errs_plain])))
        match["random_keep_aware"].append(float(np.mean([e for e, _ in errs_aware])))
        match_cos["random_keep_aware"].append(float(np.mean([c for _, c in errs_aware])))
        kept_only = [i for i, s in enumerate(sources) if s in keep]
        err, cos = gradient_error(pool.G, kept_only, pool_mean)
        match["oracle_full_pool_kept_only"].append(err)
        match_cos["oracle_full_pool_kept_only"].append(cos)
        print(f"step {t + 1}/{len(pools128)} {time.time() - t0:.0f}s", flush=True)

    k = np.array(picks_count, dtype=float)
    expect = np.array(random_expect)
    result = {
        "steps": len(pools128),
        "picks_per_step": summarise(k),
        "random_expected_overlap": summarise(expect),
        "overlap_with_oracle": {n: summarise(v) for n, v in overlap.items()},
        "skill_vs_random": {
            n: summarise((np.array(v, dtype=float) - expect) / (k - expect))
            for n, v in overlap.items()
        },
        "overlap_with_oracle_steps_ge1": {n: summarise(v[1:]) for n, v in overlap.items()},
        "pair_agreement": {k2: summarise(v) for k2, v in pair_agree.items()},
        "pair_agreement_fraction": {k2: summarise(np.array(v) / k) for k2, v in pair_agree.items()},
        "facility_distance_vs_oracle": {
            n: {kk: summarise(vv) for kk, vv in g.items()} for n, g in geo.items()
        },
        "rank1_energy_fraction_of_features": {n: summarise(v) for n, v in rank1.items()},
        "rank1_energy_fraction_centered": {n: summarise(v) for n, v in rank1c.items()},
        "skill_vs_oracle_full_l2": {
            n: summarise((np.array(v, dtype=float) - expect) / (k - expect))
            for n, v in overlap_l2.items()
        },
        "z0_mask_mean_abs_z_of_selected_coordinates": summarise(mask_z),
        "z_mean_abs_expected": float(np.sqrt(2 / np.pi)),
        "gradient_matching_relative_error": {n: summarise(v) for n, v in match.items()},
        "gradient_matching_cosine": {n: summarise(v) for n, v in match_cos.items()},
        "gradient_matching_weighted_relative_error": {n: summarise(v) for n, v in match_w.items()},
        "picks_by_norm_or_loss_overlap": {n: summarise(v) for n, v in pseudo_overlap.items()},
        "spearman_of_the_z0_feature": {n: summarise(v) for n, v in corr.items()},
        "per_step": {"overlap": overlap, "gradient_error": match},
    }
    return result


def zdraw(pools128, z0, rng, draws=24):
    """T2: the recipe's chain (fixed z over the sequence) for many independent z."""
    d = z0.numel()
    oracle = Chain("oracle", lambda t, p, c: p.G)
    keep = oracle.args.keep_source_ids
    oracle_picks, expects, ks = [], [], []
    for t, pool in enumerate(pools128):
        oracle.step(t, pool, {})
        oracle_picks.append(set(oracle.picks))
        cand = [i for i, s in enumerate(pool.sources) if s not in keep]
        labels, classes, quotas = class_budgets(
            SELECT - (len(pool) - len(cand)),
            len(cand),
            np.array([pool.sources[i] for i in cand]),
            "floor",
            "proportional",
        )
        uniq = np.unique([pool.sources[i] for i in cand])
        by_source = {int(uniq[c]): int(k) for c, k in zip(classes, quotas, strict=True)}
        expects.append(random_overlap([pool.sources[i] for i in cand], by_source))
        ks.append(SELECT - (len(pool) - len(cand)))

    def run(z_fixed):
        chain = Chain("fixed", lambda t, p, c: torch.outer(p.G @ z_fixed, z_fixed))
        skills, errs = [], []
        for t, pool in enumerate(pools128):
            chain.step(t, pool, {})
            overlap = len(set(chain.picks) & oracle_picks[t])
            skills.append((overlap - expects[t]) / (ks[t] - expects[t]))
            errs.append(gradient_error(pool.G, chain.all, pool.G.mean(0))[0])
        return float(np.mean(skills)), float(np.mean(errs))

    z0_result = run(z0)
    others = []
    for i in range(draws):
        others.append(run(draw(rng, d)))
        print(f"zdraw {i + 1}/{draws}", flush=True)
    skills = np.array([s for s, _ in others])
    errs = np.array([e for _, e in others])
    return {
        "z0": {"skill": z0_result[0], "gradient_error": z0_result[1]},
        "random_z": {"skill": summarise(skills), "gradient_error": summarise(errs)},
        "z0_skill_percentile_among_random_z": float((skills < z0_result[0]).mean() * 100),
        "z0_error_percentile_among_random_z": float((errs < z0_result[1]).mean() * 100),
    }


def mean_check(pools32, z0, rng):
    """T3: cosine of (1/m) sum_j (g.z_j) z_j with g, for m = 1, 16, 256, 4096."""
    d = z0.numel()
    G = torch.cat([p.G for p in pools32[:2]])  # 64 examples
    Gn = G / G.norm(dim=1, keepdim=True)
    result = {}
    for m in (1, 16, 256, 4096):
        cosines = []
        for _trial in range(3 if m < 4096 else 1):
            acc = torch.zeros_like(G)
            for start in range(0, m, 256):
                width = min(256, m - start)
                Z = draw(rng, d, width)
                acc += (G @ Z) @ Z.T
            est = acc / m
            cosines.append(((est * Gn).sum(1) / est.norm(dim=1)).numpy())
        cos = np.concatenate(cosines)
        result[f"m={m}"] = {
            "cosine_mean": float(np.mean(cos)),
            "cosine_std": float(np.std(cos)),
            "theory_sqrt_m_over_m_plus_d": float(np.sqrt(m / (m + d))),
            "relative_error_mean": float(
                np.mean(np.sqrt(np.maximum(1 / np.array(cos) ** 2 - 1, 0)))
            ),
        }
    return result


def weights_check(pools128, rng):
    """Global token mean (library) vs mean of per-micro-batch token means (upstream), 64 selected."""
    cos, rel, dispersion = [], [], []
    keep = TrainingArguments(output_dir="unused").keep_source_ids
    for pool in pools128:
        labels = torch.from_numpy(pool.labels.astype(np.float32))
        kept = [i for i, s in enumerate(pool.sources) if s in keep]
        cand = [i for i, s in enumerate(pool.sources) if s not in keep]
        for _ in range(20):
            chosen = kept + rng.choice(cand, SELECT - len(kept), replace=False).tolist()
            chosen = rng.permutation(chosen)
            n = labels[chosen]
            g = pool.G[chosen]
            # the gradient of the token-sum loss of example i is n_i g_i (g_i: own-length mean)
            ours = (n[:, None] * g).sum(0) / n.sum()
            pairs = chosen.reshape(-1, 2)  # micro-batches of 2, all weighted alike
            w_theirs = torch.zeros(len(chosen))
            for r, (a, b) in enumerate(pairs):
                na, nb = labels[a], labels[b]
                w_theirs[2 * r], w_theirs[2 * r + 1] = (
                    na / (na + nb) / len(pairs),
                    nb / (na + nb) / len(pairs),
                )
            theirs = (w_theirs[:, None] * g).sum(0)
            w_ours = n / n.sum()
            cos.append(float(ours @ theirs / ours.norm() / theirs.norm()))
            rel.append(float((ours - theirs).norm() / theirs.norm()))
            dispersion.append(
                (
                    float(w_ours.max() * len(chosen)),
                    float(w_theirs.max() * len(chosen)),
                    float((w_ours - w_theirs).abs().sum() / 2),
                )
            )
    d = np.array(dispersion)
    return {
        "gradient_cosine_ours_vs_upstream": summarise(cos),
        "gradient_relative_difference": summarise(rel),
        "max_example_weight_x_uniform_ours": summarise(d[:, 0]),
        "max_example_weight_x_uniform_upstream": summarise(d[:, 1]),
        "total_variation_between_weightings": summarise(d[:, 2]),
        "note": "weights on the per-example mean gradients: token share of the step (ours) or of the micro-batch pair / 32 (upstream); uniform is 1/64",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--parts", nargs="+", default=["geometry", "steps", "zdraw", "mean", "weights"]
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-pools128", type=int, default=0)
    parser.add_argument("--zdraws", type=int, default=24)
    args = parser.parse_args()
    z0, pools32, pools128 = load(args.root)
    if args.max_pools128:
        pools128 = pools128[: args.max_pools128]
    print(f"{len(pools32)} pools of 32, {len(pools128)} pools of 128, d={z0.numel()}", flush=True)
    rng = np.random.default_rng(args.seed)
    result = json.loads(args.out.read_text()) if args.out.exists() else {}
    for part in args.parts:
        t0 = time.time()
        if part == "geometry":
            result[part] = geometry(pools32, z0, rng)
        elif part == "steps":
            result[part] = steps(pools128, z0, rng)
        elif part == "zdraw":
            result[part] = zdraw(pools128, z0, rng, args.zdraws)
        elif part == "mean":
            result[part] = mean_check(pools32, z0, rng)
        elif part == "weights":
            result[part] = weights_check(pools128, rng)
        print(f"{part} done in {time.time() - t0:.0f}s", flush=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
