import torch
from conftest import add_lora

from colm.train.custom_phi import DecomposedPhiCausalLM


def _batch(tokenizer):
    texts = ["Hello world, this is CoLM.", "Short one.", "A third, slightly longer example!"]
    enc = tokenizer(texts, padding=True, return_tensors="pt")
    labels = enc.input_ids.clone()
    labels[enc.attention_mask == 0] = -100
    labels[:, :3] = -100  # masked prompt part
    return enc.input_ids, enc.attention_mask, labels


def _randomize_lora_b(model):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.05)


def test_decomposed_forward_matches_stock_phi(tokenizer, phi):
    model = add_lora(phi)
    _randomize_lora_b(model)  # LoRA B starts at zero; make the adapters matter
    model.eval()
    input_ids, attention_mask, labels = _batch(tokenizer)

    with torch.no_grad():
        ref = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        decomposer = DecomposedPhiCausalLM(model.get_base_model())
        mid = decomposer.forward_till_penultimate(
            input_ids=input_ids, attention_mask=attention_mask
        )
        loss, logits = decomposer.forward_final_layer(mid, labels=labels, per_sample_loss=False)
        per_sample, _ = decomposer.forward_final_layer(mid, labels=labels, per_sample_loss=True)

    valid = attention_mask.bool()
    torch.testing.assert_close(logits[valid], ref.logits.float()[valid], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(loss, ref.loss, rtol=1e-5, atol=1e-6)

    # Per-sample loss: mean of per-token CE over all shifted positions (ignored ones count as 0).
    shift_logits = ref.logits.float()[:, :-1]
    shift_labels = labels[:, 1:]
    per_token = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]), shift_labels.reshape(-1), reduction="none"
    ).view_as(shift_labels)
    torch.testing.assert_close(per_sample, per_token.mean(dim=1), rtol=1e-5, atol=1e-6)
    assert per_sample.shape == (input_ids.shape[0],)


def test_last_layer_perturbation_changes_only_final_part(tokenizer, phi):
    model = add_lora(phi)
    model.eval()
    input_ids, attention_mask, labels = _batch(tokenizer)
    decomposer = DecomposedPhiCausalLM(model.get_base_model())
    last_b = [p for n, p in model.named_parameters() if "layers.1.self_attn.v_proj.lora_B" in n]
    assert len(last_b) == 1
    with torch.no_grad():
        mid = decomposer.forward_till_penultimate(
            input_ids=input_ids, attention_mask=attention_mask
        )
        base_loss, _ = decomposer.forward_final_layer(mid, labels=labels)
        last_b[0].add_(0.1)
        perturbed_loss, _ = decomposer.forward_final_layer(mid, labels=labels)
        full = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        full_per_sample, _ = decomposer.forward_final_layer(
            decomposer.forward_till_penultimate(input_ids=input_ids, attention_mask=attention_mask),
            labels=labels,
            per_sample_loss=False,
        )
    assert not torch.allclose(base_loss, perturbed_loss)
    # Reusing the cached penultimate states is exact because only the last layer moved.
    torch.testing.assert_close(full_per_sample, full.loss, rtol=1e-5, atol=1e-6)
