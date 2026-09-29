"""CPU checks of the selection-precision measurement helpers (`scripts/diagnostics/precision_arms.py`)."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from equivalence.fixtures import model_fp64

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "diagnostics"))
import precision_arms as arms  # noqa: E402

from colm.data.get_training_dataset import get_training_dataset, tokenize_examples  # noqa: E402
from colm.selection import features  # noqa: E402
from colm.selection.features import MezoEfficient  # noqa: E402
from colm.selection.packing import pack  # noqa: E402
from colm.selection.select import CoresetSelector  # noqa: E402
from colm.selection.zo import LastLayerSplit, zo_parameters  # noqa: E402
from colm.train.training_arguments import TrainingArguments  # noqa: E402


@pytest.fixture()
def setup(tokenizer, mixture_file):
    dataset = get_training_dataset([mixture_file], tokenizer=tokenizer, context_length=512)
    batch = pack(tokenize_examples(tokenizer, [dataset[i] for i in range(6)]))
    model = model_fp64(tokenizer).eval()
    split = LastLayerSplit(model.get_base_model())
    params = zo_parameters(model, ["v_proj"], -1)
    names, tensors = [n for n, _ in params], [p for _, p in params]
    gen = torch.Generator().manual_seed(3)
    zs = [torch.randn(p.shape, generator=gen, dtype=p.dtype) for p in tensors]
    return SimpleNamespace(
        model=model, split=split, params=params, names=names, tensors=tensors, zs=zs, batch=batch
    )


def test_exact_derivative_equals_a_fine_finite_difference(setup):
    s = setup
    state = arms.prefix_state(s.split, s.batch, low=False)
    exact = arms.exact_estimate(s.split, state, s.batch, s.names, s.zs)
    fine = arms.fd_estimate(s.split, state, s.batch, s.names, s.tensors, s.zs, 1e-6)
    torch.testing.assert_close(exact, fine, rtol=1e-6, atol=1e-12)
    coarse = arms.fd_estimate(s.split, state, s.batch, s.names, s.tensors, s.zs, 1e-3)
    torch.testing.assert_close(exact, coarse, rtol=1e-3, atol=1e-9)


def test_double_split_promotes_a_float32_state(setup):
    s = setup
    state32 = arms.cast_state(arms.prefix_state(s.split, s.batch, low=False), torch.float32)
    split64 = arms.double_split(s.split)  # already float64: a copy with the same values
    state64 = arms.cast_state(state32, torch.float64)
    exact = arms.exact_estimate(split64, state64, s.batch, s.names, s.zs)
    reference = arms.exact_estimate(
        s.split, arms.prefix_state(s.split, s.batch, low=False), s.batch, s.names, s.zs
    )
    torch.testing.assert_close(exact, reference, rtol=1e-4, atol=1e-8)


def test_fd_estimate_is_the_library_estimate(tmp_path, setup):
    s = setup
    args = TrainingArguments(output_dir=str(tmp_path))
    extractor = MezoEfficient(args, s.model, s.params, 7)
    library = extractor.extract(s.batch)
    state = arms.prefix_state(s.split, s.batch, low=False)
    mine = arms.fd_estimate(
        s.split, state, s.batch, extractor.perturbation.names, extractor.perturbation.params,
        extractor.perturbation.z(), args.mezo_eps,
    )  # fmt: skip
    torch.testing.assert_close(mine, library, rtol=1e-9, atol=1e-12)


def test_all_half_extract_runs_with_low_precision_autocast(tmp_path, setup, monkeypatch):
    s = setup
    monkeypatch.setattr(arms, "LOW_DTYPE", torch.bfloat16)
    model = s.model.float()
    args = TrainingArguments(output_dir=str(tmp_path))
    extractor = MezoEfficient(args, model, zo_parameters(model, ["v_proj"], -1), 7)
    full = extractor.extract(s.batch)
    half = arms.all_half_extract(extractor, s.batch)
    assert half.shape == full.shape and torch.isfinite(half).all()


def test_random_selection_is_uniform_and_seeded(monkeypatch):
    monkeypatch.setattr(features.MezoEfficient, "extract", features.MezoEfficient.extract)
    monkeypatch.setattr(CoresetSelector, "__call__", CoresetSelector.__call__)
    arms.install_random_selection(5)
    selector = CoresetSelector(TrainingArguments(output_dir="unused"), 2)
    first = [selector(torch.zeros(32, 4), [0] * 32, 16, step).indices for step in range(3)]
    for indices in first:
        assert len(set(indices)) == 16 and max(indices) < 32
    arms.install_random_selection(5)
    again = [selector(torch.zeros(32, 4), [0] * 32, 16, step).indices for step in range(3)]
    assert first == again and len({tuple(i) for i in first}) == 3
    assert np.isclose(
        np.mean([len(set(a) & set(b)) for a, b in zip(first, again[::-1])]), 8, atol=4
    )


def test_measure_pack_returns_every_arm(setup, monkeypatch):
    import measure_selection_precision as measure

    monkeypatch.setattr(arms, "LOW_DTYPE", torch.bfloat16)
    s = setup
    model = s.model.float()
    split = LastLayerSplit(model.get_base_model())
    params = zo_parameters(model, ["v_proj"], -1)
    zs = [[z.float() for z in s.zs], [(2 * z).float() for z in s.zs]]
    ctx = {
        "split": split,
        "split64": arms.double_split(split),
        "names": [n for n, _ in params],
        "params": [p for _, p in params],
        "zs": zs,
        "zs64": [[z.double() for z in direction] for direction in zs],
    }
    out = measure.measure_pack(ctx, s.batch, reverse=False)
    assert set(out) == {"R", "Rfd3", "Rfd2", "F", "F_e2", "P", "P_e2", "H", "H_e2"}
    assert all(value.shape == (2, 6) for value in out.values())
    torch.testing.assert_close(out["R"][1], 2 * out["R"][0], rtol=1e-9, atol=0)
    torch.testing.assert_close(out["Rfd3"], out["R"], rtol=1e-4, atol=1e-9)
    assert torch.corrcoef(torch.stack([out["F"].flatten(), out["R"].flatten()]))[0, 1] > 0.99
    assert set(measure.measure_pack(ctx, s.batch, reverse=True)) == {"Fr", "Pr", "Hr"}


def test_keep_aware_random_selection_trains_the_kept_sources_and_the_quotas(monkeypatch):
    monkeypatch.setattr(features.MezoEfficient, "extract", features.MezoEfficient.extract)
    monkeypatch.setattr(CoresetSelector, "__call__", CoresetSelector.__call__)
    args = TrainingArguments(output_dir="unused")  # keep_sources 0_1_3_5_7_8_9_10_11_13
    arms.install_random_selection(3, keep_aware=True)
    selector = CoresetSelector(args, 2)
    sources = [0] * 6 + [1] * 3 + [2] * 12 + [4] * 5 + [6] * 6
    for step in range(5):
        chosen = selector(torch.zeros(32, 4), sources, 16, step).indices
        assert chosen == sorted(chosen) and len(set(chosen)) == 16
        assert all(i in chosen for i in range(9))  # kept sources 0 and 1
        counts = np.bincount([sources[i] for i in chosen if sources[i] not in (0, 1)])
        assert counts[2] + counts[4] + counts[6] == 7


def test_hybrid_prefix_endpoints_are_the_fp16_and_the_fp32_prefix(setup, monkeypatch):
    monkeypatch.setattr(arms, "LOW_DTYPE", torch.bfloat16)
    s = setup
    model = s.model.float()
    split = LastLayerSplit(model.get_base_model())
    fp32 = arms.prefix_state(split, s.batch, low=False)  # also verifies the split
    low = arms.cast_state(arms.prefix_state(split, s.batch, low=True), torch.float32)
    assert (fp32.args[0] - low.args[0]).abs().max() > 1e-4  # bfloat16 differs visibly
    none = arms.hybrid_prefix_state(split, s.batch, 0)
    full = arms.hybrid_prefix_state(split, s.batch, len(split.layers) - 1)
    torch.testing.assert_close(none.args[0], low.args[0], rtol=0, atol=0)
    torch.testing.assert_close(full.args[0], fp32.args[0], rtol=1e-6, atol=1e-6)
    assert not torch.is_autocast_enabled("cpu")
