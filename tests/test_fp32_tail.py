"""The fp32 tail of the fp16 selection prefix, on CPU with bf16 autocast standing in for CUDA fp16."""

import copy
from types import SimpleNamespace

import pytest
import torch
from equivalence.fixtures import add_lora, make_phi
from equivalence.helpers import make_args
from transformers import PhiForCausalLM

from colm.data.get_training_dataset import get_training_dataset, tokenize_examples
from colm.selection.features import MezoEfficient
from colm.selection.packing import label_positions, model_inputs, pack
from colm.selection.zo import LastLayerSplit, zo_parameters
from colm.train.frozen_weights import store_frozen_linears
from colm.train.training_arguments import TrainingArguments

LAYERS = 5  # 4 prefix layers + the perturbed last layer


@pytest.fixture()
def cpu_autocast(monkeypatch):
    """`torch.autocast("cuda", float16)` -> CPU bf16, and CUDA 'available' for the extractor."""
    real = torch.autocast

    def autocast(device_type, dtype=None, enabled=True, **kwargs):
        return real(
            "cpu", dtype=torch.bfloat16 if dtype == torch.float16 else dtype, enabled=enabled
        )

    monkeypatch.setattr(torch, "autocast", autocast)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


@pytest.fixture()
def setup(tokenizer, mixture_file):
    config = copy.deepcopy(make_phi(tokenizer).config)
    config.num_hidden_layers = LAYERS
    torch.manual_seed(0)
    model = add_lora(PhiForCausalLM(config))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(std=0.05)
    model = model.float().eval()
    dataset = get_training_dataset([mixture_file], tokenizer=tokenizer, context_length=512)
    batch = pack(tokenize_examples(tokenizer, [dataset[i] for i in range(6)]))
    return SimpleNamespace(model=model, batch=batch)


def _state(model, batch, tail, low=True):
    split = LastLayerSplit(model.get_base_model())
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=low):
        return split.prefix(tail=tail, device_type="cpu", **model_inputs(batch)).float()


def test_tail_endpoints_are_the_low_precision_and_the_fp32_prefix(setup):
    s = setup
    fp32 = _state(s.model, s.batch, 0, low=False).args[0]
    low = _state(s.model, s.batch, 0).args[0]
    assert (fp32 - low).abs().max() > 1e-4  # bfloat16 differs visibly
    full = _state(s.model, s.batch, LAYERS - 1).args[0]  # every prefix layer in fp32
    torch.testing.assert_close(full, fp32, rtol=1e-6, atol=1e-6)
    errors = [
        (_state(s.model, s.batch, k).args[0] - fp32).abs().max().item() for k in range(LAYERS)
    ]
    assert errors[0] > 1e-4 and errors[-1] < 1e-6
    assert errors[2] < errors[0]  # a shorter fp16 stretch is closer to fp32 (never a worse one)
    assert 0 < errors[2]


def test_the_tail_hook_and_the_autocast_switch_are_removed_after_the_forward(setup):
    s = setup
    split = LastLayerSplit(s.model.get_base_model())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        split.prefix(tail=2, device_type="cpu", **model_inputs(s.batch))
        assert torch.is_autocast_enabled("cpu")  # the enclosing autocast is restored
    assert not torch.is_autocast_enabled("cpu")
    assert all(not layer._forward_pre_hooks for layer in split.layers)
    # a failing forward leaves neither the hook nor a disabled autocast behind
    with pytest.raises(TypeError), torch.autocast("cpu", dtype=torch.bfloat16):
        split.prefix(tail=2, device_type="cpu", input_ids=1, unknown=2)
    assert all(not layer._forward_pre_hooks for layer in split.layers)
    assert not torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("tail", [-1, LAYERS])
def test_the_tail_must_fit_in_the_prefix(setup, tail):
    split = LastLayerSplit(setup.model.get_base_model())
    with pytest.raises(ValueError, match="fp32 tail"):
        split.prefix(tail=tail, device_type="cpu", **model_inputs(setup.batch))


def _extractor(model, dtype, tail):
    args = SimpleNamespace(
        mezo_eps=1e-3,
        selection_prefix_dtype=dtype,
        selection_prefix_fp32_tail=tail,
        mezo_selection="grad",
    )
    return MezoEfficient(args, model, zo_parameters(model, ["v_proj"], -1), 7)


def test_the_extractor_tail_matches_the_fp32_and_the_fp16_prefix(setup, cpu_autocast):
    s = setup
    fp32 = _extractor(s.model, "float32", 0).extract(s.batch)
    fp16 = _extractor(s.model, "float16", 0).extract(s.batch)
    assert not torch.allclose(fp16, fp32, rtol=1e-3, atol=0)  # the low precision matters
    all_tail = _extractor(s.model, "float16", LAYERS - 1).extract(s.batch)
    torch.testing.assert_close(all_tail, fp32, rtol=1e-4, atol=1e-6)
    torch.testing.assert_close(
        _extractor(s.model, "float16", 0).extract(s.batch), fp16, rtol=0, atol=0
    )
    part = _extractor(s.model, "float16", 2).extract(s.batch)
    assert part.shape == fp32.shape and torch.isfinite(part).all()
    assert all(not layer._forward_pre_hooks for layer in s.model.get_base_model().model.layers)


def test_the_extractor_rejects_a_bad_tail(setup):
    with pytest.raises(ValueError, match=rf"in \[0, {LAYERS - 1}\]"):
        _extractor(setup.model, "float16", LAYERS)
    with pytest.raises(ValueError, match="needs selection_prefix_dtype=float16"):
        _extractor(setup.model, "float32", 2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(selection_prefix_dtype="float16", selection_prefix_fp32_tail=-1), "must be >= 0"),
        (dict(selection_prefix_dtype="float32", selection_prefix_fp32_tail=2), "needs"),
    ],
)
def test_the_tail_option_is_validated_at_load(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        make_args(tmp_path, efficient_mezo=True, **overrides)
    make_args(
        tmp_path,
        efficient_mezo=True,
        selection_prefix_dtype="float16",
        selection_prefix_fp32_tail=2,
    )


# ---------------------------------------------------------------------------------------------
# Frozen Linear weights stored in the autocast dtype (bf16 on CPU stands in for fp16 on CUDA)
# ---------------------------------------------------------------------------------------------


def _low_layers(model):
    base = model.get_base_model()
    return [
        all(
            m.weight.dtype == torch.bfloat16
            for m in layer.modules()
            if isinstance(m, torch.nn.Linear) and not m.weight.requires_grad
        )
        for layer in base.model.layers
    ]


def test_only_frozen_linears_before_the_fp32_layers_are_stored_low(setup):
    model = copy.deepcopy(setup.model)
    linear_bytes = sum(
        p.numel() * (4 - 2)
        for layer in model.get_base_model().model.layers[: LAYERS - 3]
        for m in layer.modules()
        if isinstance(m, torch.nn.Linear) and not m.weight.requires_grad
        for p in m.parameters()
    )
    assert store_frozen_linears(model, torch.bfloat16, keep_last=3) == linear_bytes > 0
    assert _low_layers(model) == [True, True, False, False, False]
    for name, p in model.named_parameters():
        low = p.dtype == torch.bfloat16
        assert not (
            low and ("lora_" in name or "norm" in name or "embed" in name or "head" in name)
        )
    assert store_frozen_linears(model, torch.bfloat16, keep_last=3) == 0  # nothing left to convert
    with pytest.raises(ValueError, match="keep_last"):
        store_frozen_linears(model, torch.bfloat16, keep_last=LAYERS + 1)


def test_stored_low_weights_give_the_same_estimate_bit_for_bit(setup, cpu_autocast):
    s = setup
    reference = _extractor(s.model, "float16", 2).extract(s.batch)
    stored = copy.deepcopy(s.model)
    store_frozen_linears(stored, torch.bfloat16, keep_last=3)
    assert torch.equal(_extractor(stored, "float16", 2).extract(s.batch), reference)


def test_stored_low_weights_give_the_same_loss_and_gradients_bit_for_bit(setup, cpu_autocast):
    def loss_and_grads(model):
        model.train()
        model.zero_grad(set_to_none=True)
        positions, targets, segment = label_positions(setup.batch)
        with torch.autocast("cuda", dtype=torch.float16):  # (CPU bf16 here)
            logits = model(**model_inputs(setup.batch), logits_to_keep=positions).logits[0]
        loss = torch.nn.functional.cross_entropy(logits.float(), targets)
        loss.backward()
        grads = [p.grad.clone() for n, p in model.named_parameters() if "lora_" in n]
        return loss.detach(), grads

    model = copy.deepcopy(setup.model)
    expected = loss_and_grads(model)
    store_frozen_linears(model, torch.bfloat16, keep_last=0)
    assert _low_layers(model) == [True] * LAYERS
    loss, grads = loss_and_grads(model)
    assert torch.equal(loss, expected[0])
    assert all(torch.equal(a, b) for a, b in zip(grads, expected[1], strict=True))


def test_the_extractor_refuses_low_precision_weights_the_selection_needs_in_fp32(setup):
    model = copy.deepcopy(setup.model)
    store_frozen_linears(model, torch.bfloat16, keep_last=1)  # the fp32 tail (2) is converted too
    with pytest.raises(ValueError, match="requires fp32 model weights"):
        _extractor(model, "float16", 2)
    _extractor(model, "float16", 0)  # a plain fp16 prefix only needs the last layer in fp32
    _extractor(model, "float32", 0)  # (the fp32 prefix has no such check)


@pytest.mark.parametrize(
    ("options", "keep"),
    [
        (dict(data_selection_method="none", fp16=True), 0),
        (dict(data_selection_method="none", bf16=True), 0),
        (dict(data_selection_method="none"), None),  # no autocast
        (
            dict(
                efficient_mezo=True,
                selection_prefix_dtype="float16",
                selection_prefix_fp32_tail=2,
                fp16=True,
            ),
            3,
        ),
        (
            dict(
                efficient_mezo=True,
                selection_prefix_dtype="float16",
                selection_prefix_fp32_tail=0,
                fp16=True,
            ),
            1,
        ),
        (
            dict(efficient_mezo=True, selection_prefix_dtype="float16", bf16=True),
            None,
        ),  # fp16 prefix
        (dict(efficient_mezo=True, selection_prefix_dtype="float32", fp16=True), None),
        (
            dict(
                efficient_mezo=False,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                fp16=True,
            ),
            None,
        ),  # extractors that run the model in fp32
        (
            dict(
                efficient_mezo=True,
                selection_prefix_dtype="float16",
                fp16=True,
                frozen_base_low_precision=False,
            ),
            None,
        ),
    ],
)
def test_which_layers_keep_fp32_weights(tmp_path, options, keep):
    from colm.train.frozen_weights import fp32_layers_needed

    if options.get("data_selection_method") == "none":
        args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, **options)
    else:
        args = make_args(tmp_path, **options)
    assert fp32_layers_needed(args) == keep


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 autocast needs CUDA")
def test_stored_fp16_weights_equal_autocast_on_the_gpu_bit_for_bit(setup):
    """Run with `CUDA_VISIBLE_DEVICES=<id>` (the suite hides the GPUs): the prefix state and the
    training loss and gradients with weights stored in fp16 are those of the per-forward cast."""

    def on_gpu(model):
        return copy.deepcopy(model).cuda().eval()

    batch = {k: v for k, v in setup.batch.items() if k != "colm_meta"}
    batch["colm_meta"] = setup.batch["colm_meta"]
    batch = {
        k: (
            v.cuda()
            if torch.is_tensor(v)
            else {a: b.cuda() for a, b in v.items()}
            if isinstance(v, dict)
            else v
        )
        for k, v in batch.items()
    }

    def run(model):
        split = LastLayerSplit(model.get_base_model())
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = split.prefix(tail=2, device_type="cuda", **model_inputs(batch)).float().args[0]
        model.train().zero_grad(set_to_none=True)
        positions, targets, _ = label_positions(batch)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(**model_inputs(batch), logits_to_keep=positions).logits[0]
        loss = torch.nn.functional.cross_entropy(logits.float(), targets)
        loss.backward()
        return state, loss.detach(), [p.grad for n, p in model.named_parameters() if "lora_" in n]

    reference = on_gpu(setup.model)
    stored = on_gpu(setup.model)
    store_frozen_linears(stored, torch.float16, keep_last=3)
    state, loss, grads = run(stored)
    expected = run(reference)
    assert torch.equal(state, expected[0]) and torch.equal(loss, expected[1])
    assert all(torch.equal(a, b) for a, b in zip(grads, expected[2], strict=True))
