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
from colm.selection.packing import model_inputs, pack
from colm.selection.zo import LastLayerSplit, zo_parameters

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
