"""Phi forward split at the last decoder layer (transformers 5.x PhiModel).

CoLM's efficient zeroth-order estimate perturbs only parameters of the last
decoder layer, so the first L-1 layers are run once and their output is reused
for the +eps and -eps forwards of the last layer.
"""

import torch
from torch import nn
from transformers.masking_utils import create_causal_mask
from transformers.models.phi.modeling_phi import PhiForCausalLM


class DecomposedPhiCausalLM:
    """Runs a `PhiForCausalLM` (possibly with LoRA layers injected) in two parts."""

    def __init__(self, causal_lm: PhiForCausalLM):
        if not isinstance(causal_lm, PhiForCausalLM):
            raise TypeError(f"DecomposedPhiCausalLM needs a PhiForCausalLM, got {type(causal_lm)}")
        self.transformer = causal_lm.model
        self.lm_head = causal_lm.lm_head
        self.config = causal_lm.config
        self.layers = self.transformer.layers

    def forward_till_penultimate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict:
        """Embeddings and every decoder layer except the last one."""
        model = self.transformer
        inputs_embeds = model.embed_tokens(input_ids)
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(
            0
        )
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=None,
            position_ids=position_ids,
        )
        hidden_states = model.embed_dropout(inputs_embeds)
        position_embeddings = model.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers[:-1]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
        return {
            "hidden_states": hidden_states,
            "causal_mask": causal_mask,
            "position_ids": position_ids,
            "position_embeddings": position_embeddings,
        }

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
        hidden_states = self.layers[-1](
            intermediate["hidden_states"],
            attention_mask=intermediate["causal_mask"],
            position_ids=intermediate["position_ids"],
            position_embeddings=intermediate["position_embeddings"],
        )
        hidden_states = self.transformer.final_layernorm(hidden_states)
        logits = self.lm_head(hidden_states).float()

        if labels is None:
            return None, logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        if per_sample_loss:
            batch_size, seq_len = shift_labels.shape
            per_token = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                reduction="none",
            ).view(batch_size, seq_len)
            return per_token.mean(dim=1), logits
        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
        )
        return loss, logits
