"""Zeroth-order (MeZO) building blocks: the perturbed parameters and the last-layer split."""

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.func import functional_call

logger = logging.getLogger(__name__)

# Modules of a decoder layer that belong to the attention block; the others are in the MLP.
ATTENTION_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj"}


def zo_parameters(
    model, last_layers: list[str], layer_index: int
) -> list[tuple[str, nn.Parameter]]:
    """Parameters MeZO perturbs: the LoRA B of `last_layers` in one decoder layer (weights without LoRA)."""
    layer = model.config.num_hidden_layers - 1 if layer_index < 0 else layer_index
    suffix = ".lora_B" if hasattr(model, "peft_config") else ""
    patterns = [
        f"layers.{layer}.{'self_attn' if m in ATTENTION_MODULES else 'mlp'}.{m}{suffix}"
        for m in last_layers
    ]
    params = [(n, p) for n, p in model.named_parameters() if any(s in n for s in patterns)]
    if not params:
        raise ValueError(f"no parameter matches {patterns}")
    logger.info(f"ZO parameters: {[n for n, _ in params]} ({sum(p.numel() for _, p in params)})")
    return params


def call(module: nn.Module, overrides: dict[str, torch.Tensor] | None, *args, **kwargs):
    """`module(*args, **kwargs)` with some parameters replaced (without touching them)."""
    if not overrides:
        return module(*args, **kwargs)
    return functional_call(module, overrides, args, kwargs)


class Perturbation:
    """theta -> theta +- eps z, z ~ N(0, I) drawn from a fixed seed (the same z for every estimate).

    z comes from a private generator and the shifted parameters are new tensors handed to
    `functional_call`: neither the training RNG nor the parameters are touched.
    """

    def __init__(self, named_params, eps: float, seed: int):
        self.names = [n for n, _ in named_params]
        self.params = [p for _, p in named_params]
        self.eps, self.seed = eps, seed
        self._z = None

    def z(self) -> list[torch.Tensor]:
        if self._z is None:
            generator = torch.Generator(self.params[0].device).manual_seed(self.seed)
            self._z = [
                torch.normal(
                    0, 1, size=p.shape, device=p.device, dtype=p.dtype, generator=generator
                )
                for p in self.params
            ]
        return self._z

    def projected_grad(self, loss_fn) -> torch.Tensor:
        """(L(theta + eps z) - L(theta - eps z)) / 2 eps for `loss_fn(overrides)`."""
        steps = [z * self.eps for z in self.z()]
        plus = {n: p + s for n, p, s in zip(self.names, self.params, steps, strict=True)}
        minus = {n: p - s for n, p, s in zip(self.names, self.params, steps, strict=True)}
        return (loss_fn(plus) - loss_fn(minus)) / (2 * self.eps)

    def features(self, g: torch.Tensor, weight_grad: bool = False) -> torch.Tensor:
        """`[n, numel]` features g_i * z (times the parameter for `mezo_selection=weight_grad`)."""
        parts = []
        for p, z in zip(self.params, self.z(), strict=True):
            update = g.reshape(-1, 1) * z.reshape(1, -1)
            if weight_grad and not torch.all(p.data == 0):
                update = update * p.data.reshape(1, -1)
            parts.append(update)
        return torch.cat(parts, dim=1)


class _PrefixDone(Exception):
    """Raised by the hook on the last layer to stop the forward once its input is captured."""


def promote(value, dtype: torch.dtype):
    """Cast every floating tensor inside (nested) args / kwargs to `dtype`."""
    if isinstance(value, torch.Tensor):
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, tuple):
        return tuple(promote(item, dtype) for item in value)
    if isinstance(value, dict):
        return {key: promote(item, dtype) for key, item in value.items()}
    return value


@dataclass
class Prefix:
    args: tuple
    kwargs: dict[str, Any]

    def float(self) -> "Prefix":
        """Promote floating prefix outputs before the perturbed fp32 suffix replay."""
        return Prefix(promote(self.args, torch.float32), promote(self.kwargs, torch.float32))


class LastLayerSplit:
    """Run the decoder up to its last layer once, and the last layer (+ norm, head) repeatedly.

    The prefix is the model's own forward, stopped by a pre-hook on the last layer that captures
    the exact arguments (hidden states, mask, rotary embeddings) the model passes to it; the
    suffix replays the last layer with some of its parameters replaced (`functional_call`).
    Works for any decoder with `.layers` and one final norm module; `verify` checks the split
    against the full forward.
    """

    def __init__(self, causal_lm: nn.Module):
        self.decoder = causal_lm.base_model
        self.layers = self.decoder.layers
        self.head = causal_lm.get_output_embeddings()
        norms = [
            m
            for n, m in self.decoder.named_children()
            if n != "layers" and "norm" in type(m).__name__.lower()
        ]
        if len(norms) != 1:
            raise ValueError(f"cannot find the final norm of {type(self.decoder).__name__}")
        self.norm = norms[0]
        self.last = self.layers[-1]
        self.last_name = f"layers.{len(self.layers) - 1}."
        self._verified = False

    def relative_name(self, name: str) -> str:
        """Parameter name relative to the last layer (`...layers.31.self_attn.x` -> `self_attn.x`)."""
        return name.split(self.last_name, 1)[1]

    @contextmanager
    def _fp32_tail(self, k: int, device_type: str):
        """Run the last `k` prefix layers in fp32 inside an enclosing autocast.

        A forward pre-hook on prefix layer `n - 1 - k` (n = number of decoder layers) casts the
        floating inputs (hidden states, position embeddings, masks) to fp32 and enters
        `autocast(enabled=False)`. Both are undone on exit, so nothing leaks out of the forward.
        `k = 0` changes nothing; `k = n - 1` is an fp32 prefix (embeddings aside).
        """
        if not 0 <= k <= len(self.layers) - 1:
            raise ValueError(f"fp32 tail must be in [0, {len(self.layers) - 1}] layers, got {k}")
        if k == 0:
            yield
            return
        disabled = torch.autocast(device_type, enabled=False)
        entered = []

        def pre_hook(module, args, kwargs):
            if not entered:
                disabled.__enter__()
                entered.append(True)
            return promote(args, torch.float32), promote(kwargs, torch.float32)

        handle = self.layers[len(self.layers) - 1 - k].register_forward_pre_hook(
            pre_hook, with_kwargs=True
        )
        try:
            yield
        finally:
            handle.remove()
            if entered:
                disabled.__exit__(None, None, None)

    def prefix(self, *, tail: int = 0, device_type: str = "cuda", **inputs) -> Prefix:
        """Decoder up to the last layer, under the caller's autocast, with `tail` layers in fp32."""
        captured = {}

        def hook(module, args, kwargs):
            captured["value"] = Prefix(args, kwargs)
            raise _PrefixDone

        with self._fp32_tail(tail, device_type):
            handle = self.last.register_forward_pre_hook(hook, with_kwargs=True)
            try:
                try:
                    self.decoder(**{"use_cache": False, **inputs})
                except _PrefixDone:
                    pass
            finally:
                handle.remove()
        state = captured["value"]
        if not self._verified:
            self._verify(inputs, state, tail, device_type)
        return state

    def hidden(self, state: Prefix, overrides: dict[str, torch.Tensor] | None = None):
        """Final-norm output of the last layer replayed with `overrides` (relative names)."""
        out = call(self.last, overrides, *state.args, **state.kwargs)
        return self.norm(out[0] if isinstance(out, tuple) else out)

    def logits(self, state: Prefix, overrides=None) -> torch.Tensor:
        logits = self.head(self.hidden(state, overrides))
        return logits.to(torch.promote_types(logits.dtype, torch.float32))

    def _verify(self, inputs, state, tail: int, device_type: str) -> None:
        """The replay reproduces the forward under the same precision regime as the prefix."""
        with self._fp32_tail(tail, device_type):
            reference = self.decoder(**{"use_cache": False, **inputs}).last_hidden_state
            with torch.autocast(device_type, enabled=False) if tail else nullcontext():
                replay = self.hidden(state)
        if not torch.allclose(replay, reference, rtol=1e-4, atol=1e-5):
            raise RuntimeError("the last-layer split does not reproduce the model's forward")
        self._verified = True
