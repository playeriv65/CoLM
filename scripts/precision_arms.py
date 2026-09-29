"""Precision arms of the MeZO selection forward: measurement helpers (not library code).

The library computes the projected gradient g_i of every example as

    g_i = (L_i(B + eps z) - L_i(B - eps z)) / (2 eps)

with the decoder prefix (layers 0..30) run once and the perturbed last layer replayed twice
(`colm.selection.features.MezoEfficient`). This module re-expresses that computation with the
precision of every stage made explicit, so the arms of `docs/selection-precision.md` can be built
from one shared prefix:

* `F`  fp32 prefix, fp32 suffix (`selection_prefix_dtype=float32`);
* `P`  fp16-autocast prefix promoted to fp32, fp32 suffix (`selection_prefix_dtype=float16`);
* `H`  fp16 autocast for the prefix AND the suffix (the upstream regime; there is no library
  switch, so `all_half_extract` is patched into `MezoEfficient` by the training launcher only);
* `R`  reference: the fp32 prefix state promoted to float64, then the last layer, final norm, head
  and loss in float64 and the EXACT directional derivative d/dt L_i(B + t z) at t = 0 (forward-mode
  autodiff, no finite difference). The prefix rounding is common to L+ and L-, so it does not
  enter g_i at the level that matters here; `scripts/measure_selection_precision.py --validate`
  checks that shortcut against an all-float64 finite difference.
"""

import copy
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from colm.selection import features
from colm.selection.facility_location import class_budgets
from colm.selection.features import example_means
from colm.selection.packing import label_counts, label_positions, model_inputs
from colm.selection.select import CoresetSelector, Selection
from colm.selection.zo import Prefix

LOW_DTYPE = torch.float16  # tests on CPU set bfloat16


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def amp(device_type: str):
    return torch.autocast(device_type, dtype=LOW_DTYPE)


def promote(value, dtype):
    """Cast every floating tensor inside (nested) args / kwargs to `dtype`."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(promote(item, dtype) for item in value)
    if isinstance(value, dict):
        return {key: promote(item, dtype) for key, item in value.items()}
    return value


def cast_state(state: Prefix, dtype) -> Prefix:
    return Prefix(promote(state.args, dtype), promote(state.kwargs, dtype))


def prefix_state(split, pack: dict, low: bool) -> Prefix:
    """The decoder prefix of a packed batch, in fp32 (`low=False`) or under fp16 autocast."""
    device_type = pack["input_ids"].device.type
    with torch.no_grad(), amp(device_type) if low else nullcontext():
        return split.prefix(**model_inputs(pack))


@contextmanager
def fp32_tail(split, k: int, device_type: str):
    """Inside an fp16-autocast prefix: the last `k` layers of the prefix run in fp32.

    A forward pre-hook on prefix layer `n - 1 - k` (n = number of decoder layers) casts the floating
    inputs (hidden states, position embeddings, masks) to fp32 and enters `autocast(enabled=False)`,
    which stays active until the prefix has stopped (the last layer's own pre-hook ends the forward).
    `k = 0` is the plain fp16 prefix, `k = n - 1` an fp32 prefix.
    """
    if k <= 0:
        yield
        return
    layer = len(split.layers) - 1 - k
    disabled = torch.autocast(device_type, enabled=False)
    entered = []

    def pre_hook(module, args, kwargs):
        disabled.__enter__()
        entered.append(True)
        return promote(args, torch.float32), promote(kwargs, torch.float32)

    handle = split.layers[layer].register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
        if entered:
            disabled.__exit__(None, None, None)


def hybrid_prefix_state(split, pack: dict, k: int) -> Prefix:
    """Prefix in fp16 autocast with an fp32 tail of `k` layers, promoted to fp32 (arm Pk)."""
    device_type = pack["input_ids"].device.type
    with torch.no_grad(), amp(device_type), fp32_tail(split, k, device_type):
        state = split.prefix(**model_inputs(pack))
    return cast_state(state, torch.float32)


def suffix_losses(split, state: Prefix, pack: dict, overrides: dict) -> torch.Tensor:
    """Per-example mean label loss of the last layer replayed with `overrides` (relative names)."""
    positions, targets, segment = label_positions(pack)
    hidden = split.hidden(state, overrides)[0, positions]
    return example_means(split.head(hidden), targets, segment, label_counts(pack))


def fd_estimate(split, state, pack, names, params, zs, eps, low_suffix=False) -> torch.Tensor:
    """(L(B + eps z) - L(B - eps z)) / 2 eps, the arithmetic of `Perturbation.projected_grad`."""
    device_type = pack["input_ids"].device.type
    steps = [z * eps for z in zs]
    values = []
    for sign in (1, -1):
        overrides = {
            split.relative_name(n): p.detach() + sign * s
            for n, p, s in zip(names, params, steps, strict=True)
        }
        with torch.no_grad(), amp(device_type) if low_suffix else nullcontext():
            values.append(suffix_losses(split, state, pack, overrides))
    return (values[0] - values[1]) / (2 * eps)


def double_split(split):
    """A `LastLayerSplit` whose last layer, final norm and head are float64 copies."""
    clone = copy.copy(split)
    clone.last = copy.deepcopy(split.last).double()
    clone.norm = copy.deepcopy(split.norm).double()
    clone.head = copy.deepcopy(split.head).double()
    return clone


def exact_estimate(split64, state64, pack, names, zs64) -> torch.Tensor:
    """Exact d/dt L_i(B + t z) at t = 0 for every example (float64, forward-mode autodiff)."""
    own = dict(split64.last.named_parameters())
    base = {split64.relative_name(n): own[split64.relative_name(n)].detach() for n in names}
    tangent = {split64.relative_name(n): z for n, z in zip(names, zs64, strict=True)}

    def loss(t):
        overrides = {k: v + t * tangent[k] for k, v in base.items()}
        return suffix_losses(split64, state64, pack, overrides)

    zero = torch.zeros((), dtype=torch.float64, device=pack["input_ids"].device)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        return torch.func.jvp(loss, (zero,), (torch.ones_like(zero),))[1]


def all_half_extract(self, pack):
    """`MezoEfficient.extract` with fp16 autocast over the prefix and the suffix (regime H)."""
    positions, targets, segment = label_positions(pack)
    counts = label_counts(pack)
    with torch.inference_mode(), amp(pack["input_ids"].device.type):
        state = self.split.prefix(**model_inputs(pack))

        def loss(overrides):
            overrides = {self.names[n]: v for n, v in overrides.items()}
            hidden = self.split.hidden(state, overrides)[0, positions]
            return example_means(self.split.head(hidden), targets, segment, counts)

        return self.perturbation.projected_grad(loss)


def install_all_half():
    features.MezoEfficient.extract = all_half_extract


def install_random_selection(seed: int, keep_aware: bool = False):
    """Random `total` of the pool per step; the MeZO forward is skipped (zeros).

    `keep_aware=False`: uniform 16 of 32, the kept sources get no special treatment.
    `keep_aware=True`: the structure of the selector with a random ranking: every kept-source
    example is trained, the remaining budget is spread over the other sources with the selector's
    per-source quotas and drawn uniformly inside each source.
    The training data, the pools, the number of selected examples and the training compute are
    those of the selection runs; only which examples are chosen differs.
    """
    rng = np.random.default_rng(seed)

    def zero_extract(self, pack):
        return torch.zeros(len(label_counts(pack)), device=pack["input_ids"].device)

    def random_call(self, feats, sources, total, step):
        if not keep_aware:
            chosen = rng.choice(len(feats), size=total, replace=False).tolist()
        else:
            args = self.args
            sources = np.asarray(sources)
            is_kept = np.isin(sources, args.keep_source_ids)
            kept, free = np.flatnonzero(is_kept), np.flatnonzero(~is_kept)
            chosen = kept[:total].tolist()
            budget = total - len(kept)
            if budget > 0:
                labels, classes, quotas = class_budgets(
                    budget, len(free), sources[free], args.num_per_class_start, "proportional"
                )
                for c, quota in zip(classes, quotas, strict=True):
                    members = free[labels == c]
                    chosen += rng.choice(members, size=int(quota), replace=False).tolist()
        chosen = sorted(chosen)
        return Selection(chosen, [1.0] * total, list(range(len(feats))))

    features.MezoEfficient.extract = zero_extract
    CoresetSelector.__call__ = random_call
