"""Phi forward split at the last decoder layer (transformers 5.x PhiModel).

CoLM's efficient zeroth-order estimate perturbs only parameters of the last
decoder layer, so the first L-1 layers are run once and their output is reused
for the +eps and -eps forwards of the last layer.
"""

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.models.phi.modeling_phi import PhiForCausalLM

from colm.train.step_timing import StepTimer


def _at_least_fp32(logits: torch.Tensor) -> torch.Tensor:
    """Upcast half-precision logits for the loss (fp32 / fp64 stay as they are)."""
    return logits if logits.dtype in (torch.float32, torch.float64) else logits.float()


class DecomposedPhiCausalLM:
    """Runs a `PhiForCausalLM` (possibly with LoRA layers injected) in two parts."""

    def __init__(self, causal_lm: PhiForCausalLM, timer: StepTimer | None = None):
        if not isinstance(causal_lm, PhiForCausalLM):
            raise TypeError(f"DecomposedPhiCausalLM needs a PhiForCausalLM, got {type(causal_lm)}")
        self.transformer = causal_lm.model
        self.lm_head = causal_lm.lm_head
        self.config = causal_lm.config
        self.layers = self.transformer.layers
        self.timer = timer or StepTimer()
        self._time_sub_blocks = False
        if self.timer.fine_enabled:
            self._add_sub_block_hooks(self.layers[-1])

    def _add_sub_block_hooks(self, layer):
        """Time the parallel sub-blocks of the last layer (only inside forward_final_layer)."""
        for name in ("input_layernorm", "self_attn", "mlp"):

            def pre(module, args, name=name):
                if self._time_sub_blocks:
                    self.timer.start(name)

            def post(module, args, output, name=name):
                if self._time_sub_blocks:
                    self.timer.stop(name)

            getattr(layer, name).register_forward_pre_hook(pre)
            getattr(layer, name).register_forward_hook(post)

    def forward_till_penultimate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        attention_kwargs: dict | None = None,
    ) -> dict:
        """Embeddings and every decoder layer except the last one (no KV cache).

        Packed inputs (colm.train.packing) pass restarting `position_ids`, no
        `attention_mask` and, for varlen kernels, `attention_kwargs` (`cu_seq_lens_q`, ...).
        """
        model, t = self.transformer, self.timer
        attention_kwargs = attention_kwargs or {}
        with t.fine("embed"):
            inputs_embeds = model.embed_tokens(input_ids)
        with t.fine("position_ids"):
            if position_ids is None:
                position_ids = torch.arange(
                    inputs_embeds.shape[1], device=inputs_embeds.device
                ).unsqueeze(0)
        with t.fine("causal_mask"):
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
            )
        with t.fine("embed_dropout"):
            hidden_states = model.embed_dropout(inputs_embeds)
        with t.fine("rotary"):
            position_embeddings = model.rotary_emb(hidden_states, position_ids=position_ids)
        with t.fine("layers"):
            for i, decoder_layer in enumerate(self.layers[:-1]):
                with t.fine(f"{i:02d}"):
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                        **attention_kwargs,
                    )
        return {
            "hidden_states": hidden_states,
            "causal_mask": causal_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
            "attention_kwargs": attention_kwargs,
        }

    def _last_layer(self, intermediate: dict) -> torch.Tensor:
        return self.layers[-1](
            intermediate["hidden_states"],
            attention_mask=intermediate["causal_mask"],
            position_ids=intermediate["position_ids"],
            position_embeddings=intermediate["position_embeddings"],
            **intermediate.get("attention_kwargs", {}),
        )

    def final_layer_token_losses(
        self,
        intermediate: dict,
        targets: torch.LongTensor,
        positions: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """Per-position cross-entropy of the last decoder layer + LM head (fp32).

        Without `positions`, `targets` is `[rows, T]` (the token predicted at every position,
        -100 = ignored, loss 0) and the result has that shape. With `positions` (flat indices
        into `rows * T`) only those positions go through the final layer norm and the LM head;
        `targets` and the result are aligned with `positions`.
        """
        t = self.timer
        with t.fine("decoder_layer"):
            self._time_sub_blocks = t.fine_enabled
            try:
                hidden_states = self._last_layer(intermediate)
            finally:
                self._time_sub_blocks = False
        if positions is not None:
            with t.fine("label_positions"):
                hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])[positions]
        with t.fine("final_layernorm"):
            hidden_states = self.transformer.final_layernorm(hidden_states)
        with t.fine("lm_head"):
            logits = _at_least_fp32(self.lm_head(hidden_states))
        with t.fine("cross_entropy"):
            losses = nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            )
        return losses.view(targets.shape)

    def forward_final_layer(
        self,
        intermediate: dict,
        labels: torch.LongTensor | None = None,
        per_sample_loss: bool = True,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Last decoder layer, final layer norm, LM head and loss.

        Returns `(loss, logits)`. With `per_sample_loss` the loss has shape `[batch]`
        and is the mean of the per-token losses over the whole (padded) sequence,
        ignored positions contributing zero; otherwise it is the usual token mean.
        """
        t = self.timer
        with t.fine("decoder_layer"):
            self._time_sub_blocks = t.fine_enabled
            try:
                hidden_states = self._last_layer(intermediate)
            finally:
                self._time_sub_blocks = False
        with t.fine("final_layernorm"):
            hidden_states = self.transformer.final_layernorm(hidden_states)
        with t.fine("lm_head"):
            logits = self.lm_head(hidden_states)
        with t.fine("logits_float"):
            logits = _at_least_fp32(logits)

        if labels is None:
            return None, logits
        with t.fine("shift_contiguous"):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        if per_sample_loss:
            batch_size, seq_len = shift_labels.shape
            with t.fine("cross_entropy"):
                per_token = nn.functional.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(batch_size, seq_len)
            with t.fine("per_sample_mean"):
                per_sample = per_token.mean(dim=1)
            return per_sample, logits
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
        )
        return loss, logits
