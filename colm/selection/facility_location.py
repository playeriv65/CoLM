"""Source-wise facility-location selection (submodlib) with per-source budgets."""

import logging

import numpy as np
import torch
from submodlib import FacilityLocationFunction

logger = logging.getLogger(__name__)


def similarity(X, metric: str) -> np.ndarray:
    """Pairwise similarity `[N, N]` of the rows of X (float32); `l1` / `euclidean` as max - distance."""
    X = torch.as_tensor(X).to(torch.float32)
    if metric == "cosine":
        Xn = X / torch.norm(X, p=2, dim=1).unsqueeze(1)
        S = Xn @ Xn.T
    elif metric in ("l1", "euclidean"):
        dists = torch.cdist(X, X, p=1 if metric == "l1" else 2)
        S = dists.max() - dists
    else:
        raise ValueError(f"unknown metric: {metric}")
    if torch.isnan(S).any():
        logger.warning("NaN in the similarity matrix; the affected examples are not selected")
        S = torch.nan_to_num(S, nan=-0.95)
    return S.cpu().numpy()


def _fill_to(counts: np.ndarray, budget: int, order: np.ndarray) -> np.ndarray:
    """Add one to the entries of `counts` in the given order (cyclically) until they sum to budget."""
    counts = counts.copy()
    for i in range(budget - counts.sum()):
        counts[order[i % len(counts)]] += 1
    return counts


def _budgets(budget: int, y: np.ndarray, classes: np.ndarray, start: str, strategy: str):
    """Number of examples selected from every class (source)."""
    n = len(y)
    sizes = np.array([(y == c).sum() for c in classes])
    if strategy == "none":
        return np.int32([budget])
    if strategy == "balanced":
        low = np.floor(sizes / n * budget).astype(np.int32)
        high = np.ceil(sizes / n * budget).astype(np.int32)
        free = np.random.permutation(np.where(low != high)[0])
        counts = low.copy()
        for i in range(budget - low.sum()):
            counts[free[i % len(low)]] += 1
        return counts
    if strategy == "proportional":
        if start == "floor":
            counts = np.floor(sizes / n * budget).astype(np.int32)
            return _fill_to(counts, budget, np.argsort(counts))
        counts = np.ceil(sizes / n * budget).astype(np.int32)
        for i in range(counts.sum() - budget):
            j = np.argsort(counts)[::-1][i % len(counts)]
            if counts[j] <= 0:
                raise ValueError("cannot decrease the per-class budget below zero")
            counts[j] -= 1
        return counts
    raise ValueError(f"unknown strategy: {strategy}")


def get_orders_and_weights(B, X, metric, y=None, per_class_start="floor", strategy="proportional"):
    """Select `B` of the rows of X by facility location, `B_c` from each class of `y`.

    Returns `(order, weights)`: indices into X (classes concatenated, greedy order inside a class)
    and the size of the cluster each selected example represents.
    """
    if y is None:
        if strategy != "none":
            raise ValueError(f"strategy {strategy!r} needs class labels")
        y = np.zeros(len(X), dtype=np.int32)
    else:
        y = np.unique(np.asarray(y), return_inverse=True)[1]
    classes = np.unique(y)
    counts = _budgets(B, y, classes, per_class_start, strategy)
    assert counts.sum() == B

    order, weights = [], []
    for c in classes:
        members = np.where(y == c)[0]
        k = counts[c]
        if k == 0:
            continue
        if len(members) == 1 or len(members) == k:
            order.append(members)
            weights.append(np.ones(len(members), dtype=np.float32))
            continue
        S = similarity(X[members], metric)
        flf = FacilityLocationFunction(
            n=len(members), sijs=S, separate_rep=False, mode="dense", metric=metric
        )
        greedy = flf.maximize(
            budget=int(k),
            optimizer="LazyGreedy",
            stopIfZeroGain=False,
            stopIfNegativeGain=False,
            show_progress=False,
        )
        picked = np.array([g[0] for g in greedy], dtype=np.int32)
        # Every example is represented by its most similar selected one (itself if selected).
        owner = np.argmax(S[:, picked], axis=1)
        owner[picked] = np.arange(len(picked))
        order.append(members[picked])
        weights.append(np.bincount(owner, minlength=len(picked)).astype(np.float32))
    return np.concatenate(order).astype(np.int32), np.concatenate(weights).astype(np.float32)
