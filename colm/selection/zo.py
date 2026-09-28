"""Zeroth-order (MeZO) building blocks: the perturbed parameters and the last-layer split."""

import logging
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

    `legacy` (upstream errors E3, E-drift): reseed the *global* RNG for every draw and shift the
    parameters in place (+eps, -2 eps, +eps), so the training RNG restarts from the same state
    after every estimate and the parameters accumulate rounding error.
    """

    def __init__(self, named_params, eps: float, seed: int, legacy: bool = False):
        self.names = [n for n, _ in named_params]
        self.params = [p for _, p in named_params]
        self.eps, self.seed, self.legacy = eps, seed, legacy
        self._z = None

    def _draw(self) -> list[torch.Tensor]:
        if self.legacy:
            torch.manual_seed(self.seed)
            generator = None
        else:
            generator = torch.Generator(self.params[0].device).manual_seed(self.seed)
        return [
            torch.normal(0, 1, size=p.shape, device=p.device, dtype=p.dtype, generator=generator)
            for p in self.params
        ]

    def z(self) -> list[torch.Tensor]:
        if self.legacy:
            return self._draw()
        if self._z is None:
            self._z = self._draw()
        return self._z

    def projected_grad(self, loss_fn) -> torch.Tensor:
        """(L(theta + eps z) - L(theta - eps z)) / 2 eps for `loss_fn(overrides)`."""
        if self.legacy:
            self._shift(1)
            loss_plus = loss_fn(None)
            self._shift(-2)
            loss_minus = loss_fn(None)
            self._shift(1)
        else:
            z = self.z()
            plus = {n: p + s for n, p, s in zip(self.names, self.params, self._steps(z, 1))}
            minus = {n: p - s for n, p, s in zip(self.names, self.params, self._steps(z, 1))}
            loss_plus, loss_minus = loss_fn(plus), loss_fn(minus)
        return (loss_plus - loss_minus) / (2 * self.eps)

    def _steps(self, z, sign):
        return [sign * zi * self.eps for zi in z]

    def _shift(self, sign: int) -> None:
        for p, z in zip(self.params, self._draw()):
            p.data = p.data + sign * z * self.eps

    def features(self, g: torch.Tensor, weight_grad: bool = False) -> torch.Tensor:
        """`[n, numel]` features g_i * z (times the parameter for `mezo_selection=weight_grad`)."""
        if self.legacy:
            torch.manual_seed(self.seed)  # the estimate ends with a fresh draw of z
        parts = []
        for p, z in zip(self.params, self.z()):
            update = g.reshape(-1, 1) * z.reshape(1, -1)
            if weight_grad and not torch.all(p.data == 0):
                update = update * p.data.reshape(1, -1)
            parts.append(update)
        return torch.cat(parts, dim=1)


class _PrefixDone(Exception):
    """Raised by the hook on the last layer to stop the forward once its input is captured."""


@dataclass
class Prefix:
    args: tuple
    kwargs: dict[str, Any]


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

    def prefix(self, input_ids, attention_mask) -> Prefix:
        captured = {}

        def hook(module, args, kwargs):
            captured["value"] = Prefix(args, kwargs)
            raise _PrefixDone

        handle = self.last.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            try:
                self.decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            except _PrefixDone:
                pass
        finally:
            handle.remove()
        state = captured["value"]
        if not self._verified:
            self._verify(input_ids, attention_mask, state)
        return state

    def hidden(self, state: Prefix, overrides: dict[str, torch.Tensor] | None = None):
        """Final-norm output of the last layer replayed with `overrides` (relative names)."""
        out = call(self.last, overrides, *state.args, **state.kwargs)
        return self.norm(out[0] if isinstance(out, tuple) else out)

    def logits(self, state: Prefix, overrides=None) -> torch.Tensor:
        logits = self.head(self.hidden(state, overrides))
        return logits.to(torch.promote_types(logits.dtype, torch.float32))

    def _verify(self, input_ids, attention_mask, state) -> None:
        reference = self.decoder(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state
        replay = self.hidden(state)
        if not torch.allclose(replay, reference, rtol=1e-4, atol=1e-5):
            raise RuntimeError("the last-layer split does not reproduce the model's forward")
        self._verified = True


def per_sample_loss(
    logits: torch.Tensor, labels: torch.Tensor, legacy: bool = False
) -> torch.Tensor:
    """Loss of every example of a batch, `[batch]`.

    Mean over the label tokens of the example, the loss training minimises. `legacy` (upstream
    error E2): divide by the padded width of the batch minus one instead, so an example's loss
    depends on the other examples of its micro-batch.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
    batch, width = shift_labels.shape
    token_loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1), reduction="none"
    ).view(batch, width)
    if legacy:
        return token_loss.mean(dim=1)
    return token_loss.sum(dim=1) / (shift_labels != -100).sum(dim=1).clamp(min=1)
