import numpy as np
import pytest
import torch

from colm.selection.facility_location import get_orders_and_weights, similarity


@pytest.mark.parametrize("metric", ["l1", "euclidean", "cosine"])
def test_selection_without_sources(metric):
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.normal(size=(40, 16)).astype(np.float32))
    order, weights = get_orders_and_weights(10, X, metric, strategy="none")
    assert order.shape == (10,) and weights.shape == (10,)
    assert len(set(order.tolist())) == 10
    assert order.min() >= 0 and order.max() < 40
    # Every point is assigned to one selected medoid.
    assert weights.sum() == pytest.approx(40)
    assert (weights >= 1).all()


def test_proportional_source_wise_selection():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(48, 8)).astype(np.float32)
    y = np.array([0] * 24 + [5] * 16 + [9] * 8)  # arbitrary source ids
    order, weights = get_orders_and_weights(12, X, "l1", y=y, strategy="proportional")
    assert len(order) == 12 and len(set(order.tolist())) == 12
    counts = {s: int((y[order] == s).sum()) for s in np.unique(y)}
    assert counts == {0: 6, 5: 4, 9: 2}
    assert weights.sum() == pytest.approx(48)


def test_facility_location_picks_cluster_representatives():
    # Three tight, well separated clusters; a budget of three must hit each once.
    rng = np.random.default_rng(2)
    centers = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])
    X = np.concatenate([c + rng.normal(scale=0.1, size=(10, 2)) for c in centers]).astype(
        np.float32
    )
    order, weights = get_orders_and_weights(3, X, "euclidean", strategy="none")
    assert sorted(o // 10 for o in order.tolist()) == [0, 1, 2]
    assert sorted(weights.tolist()) == [10, 10, 10]


def test_similarity_handles_nan():
    X = torch.tensor([[1.0, 0.0], [float("nan"), 1.0], [0.0, 1.0]])
    assert not np.isnan(similarity(X, "cosine")).any()


def _cluster_sizes_reference(S, orders):
    """The upstream loop: every example is represented by the most similar selected one."""
    weights = np.zeros(len(orders), dtype=np.float32)
    for i in range(len(S)):
        if i in orders:
            weights[np.where(orders == i)[0][0]] += 1
        else:
            weights[np.argmax(S[i, orders])] += 1
    return weights


@pytest.mark.parametrize("integers", [False, True])  # integers: many exactly equal distances
def test_cluster_sizes_match_the_reference_loop(integers):
    rng = np.random.default_rng(3)
    X = rng.integers(0, 3, size=(30, 4)) if integers else rng.normal(size=(30, 4))
    X = torch.from_numpy(X.astype(np.float32))
    order, weights = get_orders_and_weights(8, X, "l1", strategy="none")
    np.testing.assert_array_equal(weights, _cluster_sizes_reference(similarity(X, "l1"), order))
