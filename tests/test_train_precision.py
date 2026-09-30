"""The fp32 tail of the training forward, on CPU with bf16 autocast standing in for CUDA fp16."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from equivalence.fixtures import add_lora, make_phi
from equivalence.helpers import build, make_args
from transformers import PhiForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from colm.data.get_training_dataset import get_training_dataset, tokenize_examples
from colm.selection.packing import label_positions, model_inputs, pack
from colm.train import precision
from colm.train.config import parse_args
from colm.train.frozen_weights import store_frozen_linears
from colm.train.precision import TrainingPrecision, block_mask, check_tail

LAYERS = 5
KEY = "sdpa"


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
    dataset = get_training_dataset([mixture_file], tokenizer=tokenizer, context_length=512)
    batch = pack(tokenize_examples(tokenizer, [dataset[i] for i in range(6)]))
    return SimpleNamespace(model=model.float().train(), batch=batch)


def run(setup, tail, autocast=True):
    """(logits, flat LoRA gradient) of one training forward + backward with the tail `tail`."""
    model, batch = setup.model, setup.batch
    tp = TrainingPrecision(model, tail)
    model.zero_grad(set_to_none=True)
    positions, targets, _ = label_positions(batch)
    with tp.installed(KEY), tp.running():
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            logits = model(**model_inputs(batch), logits_to_keep=positions).logits[0]
        F.cross_entropy(logits.float(), targets).backward()
    grads = torch.cat([p.grad.flatten() for n, p in model.named_parameters() if "lora_" in n])
    return logits.detach().float(), grads


def rel_error(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_all_fp32_tail_gradient_equals_the_fp32_gradient(setup):
    """No autocast and the tail on every layer: the dispatcher (MATH sdpa, block mask from
    cu_seq_lens_q) reproduces the stock fp32 attention."""
    reference = run(setup, 0, autocast=False)
    tail = run(setup, LAYERS, autocast=False)
    torch.testing.assert_close(tail[0], reference[0], rtol=1e-5, atol=1e-5)
    assert rel_error(tail[1], reference[1]) < 1e-4


def test_the_gradient_gets_closer_to_fp32_with_the_tail(setup):
    """Very sharp attention (large q, k) makes their rounding the dominant error, as in Phi-2's last
    blocks, which the fp32 tail then removes. (In this tiny model the fp16 rounding of every other
    op is a floor of about 1 percent, so the effect only shows with the tail on all layers.)"""
    with torch.no_grad():
        for name, p in setup.model.named_parameters():
            if name.endswith(("q_proj.base_layer.weight", "k_proj.base_layer.weight")):
                p.mul_(20)
    reference = run(setup, 0, autocast=False)[1]
    assert rel_error(run(setup, LAYERS)[1], reference) < 0.8 * rel_error(
        run(setup, 0)[1], reference
    )


def test_tail_zero_is_the_current_path_bit_for_bit(setup):
    before = ALL_ATTENTION_FUNCTIONS[KEY]
    tp = TrainingPrecision(setup.model, 0)
    with tp.installed(KEY):
        assert ALL_ATTENTION_FUNCTIONS[KEY] is before  # nothing registered
        assert all("forward" not in m.__dict__ for m in setup.model.modules())
    plain = run(setup, 0)
    again = run(setup, 0)
    assert torch.equal(plain[0], again[0]) and torch.equal(plain[1], again[1])


def test_projections_and_attention_run_in_fp32_only_in_the_tail(setup, monkeypatch):
    calls = []
    real = precision.fp32_attention
    monkeypatch.setattr(
        precision,
        "fp32_attention",
        lambda q, k, v, *a: (calls.append(q.dtype), real(q, k, v, *a))[1],
    )
    seen = {}
    modules = dict(setup.model.named_modules())
    for name, module in modules.items():
        if name.endswith(("q_proj", "k_proj")):
            module.register_forward_hook(
                lambda m, args, out, name=name: seen.setdefault(name, []).append((args[0], out))
            )
    run(setup, 2)
    assert calls == [torch.float32] * 2
    for name, ((x, out),) in seen.items():
        layer = int(name.split(".layers.")[1].split(".")[0])
        if layer >= LAYERS - 2:
            # fp32 in, fp32 out, and exactly what the unwrapped module gives on that input in fp32
            module = modules[name]
            assert x.dtype == out.dtype == torch.float32
            assert torch.equal(type(module).forward(module, x), out)
        else:
            assert out.dtype == torch.bfloat16  # the base projections stay under autocast


def test_the_dispatcher_falls_back_without_cu_seq_lens(setup, tokenizer, monkeypatch):
    """A padded batch (in-training evaluation) has no cu_seq_lens_q: the wrapped function runs."""
    calls = []
    wrapped = ALL_ATTENTION_FUNCTIONS[KEY]
    monkeypatch.setattr(precision, "fp32_attention", lambda *a: pytest.fail("fp32 path used"))
    counting = lambda *a, **k: (calls.append(1), wrapped(*a, **k))[1]  # noqa: E731
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, KEY, counting)
    ids = torch.randint(3, 30, (2, 12))
    mask = torch.ones_like(ids)
    mask[1, 8:] = 0
    tp = TrainingPrecision(setup.model, LAYERS)
    with tp.installed(KEY), tp.running(), torch.no_grad():
        out = setup.model(input_ids=ids, attention_mask=mask).logits
    assert len(calls) == LAYERS and torch.isfinite(out).all()


def test_the_tail_is_off_outside_running_and_removed_afterwards(setup):
    model = setup.model
    baseline = model.eval()(**model_inputs(setup.batch)).logits.detach()
    original = ALL_ATTENTION_FUNCTIONS[KEY]
    tp = TrainingPrecision(model, 3)
    projections = [m for m in model.modules() if m in tp.projections]
    with tp.installed(KEY):
        assert ALL_ATTENTION_FUNCTIONS[KEY] is not original
        assert all("forward" in m.__dict__ for m in projections) and len(projections) == 6
        # evaluation / the selection forward: not running, so the stock path, bit for bit
        assert torch.equal(model(**model_inputs(setup.batch)).logits.detach(), baseline)
        with tp.running():
            assert not torch.equal(model(**model_inputs(setup.batch)).logits.detach(), baseline)
        assert torch.equal(model(**model_inputs(setup.batch)).logits.detach(), baseline)
    assert ALL_ATTENTION_FUNCTIONS[KEY] is original
    assert all("forward" not in m.__dict__ for m in model.modules())
    # a failing training leaves nothing behind either
    with pytest.raises(RuntimeError, match="boom"), tp.installed(KEY), tp.running():
        raise RuntimeError("boom")
    assert ALL_ATTENTION_FUNCTIONS[KEY] is original
    assert all("forward" not in m.__dict__ for m in model.modules())
    assert not tp._active and tp._mask is None


def test_block_mask_is_causal_inside_each_example():
    mask = block_mask(torch.tensor([0, 2, 5]), 5, torch.device("cpu"))[0, 0]
    expected = torch.zeros(5, 5, dtype=torch.bool)
    expected[:2, :2] = torch.tril(torch.ones(2, 2)).bool()
    expected[2:, 2:] = torch.tril(torch.ones(3, 3)).bool()
    assert torch.equal(mask, expected)


def test_fp32_attention_matches_a_float64_reference_with_grouped_kv():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 7, 8, dtype=torch.bfloat16)
    k, v = (torch.randn(1, 2, 7, 8, dtype=torch.bfloat16) for _ in range(2))
    mask = block_mask(torch.tensor([0, 3, 7]), 7, torch.device("cpu"))
    out = precision.fp32_attention(q, k, v, mask, 8**-0.5)
    assert out.dtype == torch.float32 and out.shape == (1, 7, 4, 8)
    kk, vv = (x.double().repeat_interleave(2, dim=1) for x in (k, v))
    scores = (q.double() @ kk.transpose(-1, -2)) * 8**-0.5
    weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    expected = (weights @ vv).transpose(1, 2)
    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-5)


def test_check_tail_needs_fp32_qk_weights_and_a_tail_that_fits(setup):
    check_tail(setup.model, 0)
    check_tail(setup.model, LAYERS)
    with pytest.raises(ValueError, match="train_fp32_tail must be in"):
        check_tail(setup.model, LAYERS + 1)
    stored = copy.deepcopy(setup.model)
    store_frozen_linears(stored, torch.bfloat16, keep_last=1)
    check_tail(stored, 1)
    with pytest.raises(ValueError, match="needs fp32 weights.*layers.3.self_attn.q_proj"):
        check_tail(stored, 2)
    store_frozen_linears(stored, torch.bfloat16, keep_last=0, keep_qk_last=0)  # nothing left to do


def test_frozen_weights_keep_qk_of_the_tail_fp32_and_convert_the_rest(setup):
    model = copy.deepcopy(setup.model)
    store_frozen_linears(model, torch.bfloat16, keep_last=1, keep_qk_last=3)
    layers = model.get_base_model().model.layers

    def dtypes(layer, name):
        module = getattr(layer.self_attn, name)
        return {p.dtype for n, p in module.named_parameters() if "lora_" not in n}

    for i, layer in enumerate(layers):
        expect_qk = torch.float32 if i >= LAYERS - 3 else torch.bfloat16
        expect_rest = torch.float32 if i >= LAYERS - 1 else torch.bfloat16
        assert dtypes(layer, "q_proj") == dtypes(layer, "k_proj") == {expect_qk}
        assert dtypes(layer, "v_proj") == {expect_rest}
        assert layer.mlp.fc1.base_layer.weight.dtype == expect_rest
    check_tail(model, 3)
    with pytest.raises(ValueError, match="keep_qk_last"):
        store_frozen_linears(model, torch.bfloat16, keep_last=0, keep_qk_last=LAYERS + 1)


def test_the_trainer_installs_the_tail_only_while_training(
    tmp_path, tokenizer, mixture_file, monkeypatch
):
    calls = []
    real = precision.fp32_attention
    monkeypatch.setattr(
        precision, "fp32_attention", lambda q, *a: (calls.append(q.dtype), real(q, *a))[1]
    )
    args = make_args(
        tmp_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        data_selection_method="none",
        max_steps=2,
        bf16=True,
        train_fp32_tail=1,
    )
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    trainer, model = build(args, tokenizer, mixture_file)
    assert trainer.describe()["train_fp32_tail"] == 1
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original  # installed by train(), not before
    trainer.train()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    assert all("forward" not in m.__dict__ for m in trainer.precision.projections)
    assert calls and set(calls) == {torch.float64}  # the fixture model is float64
    calls.clear()
    with torch.no_grad():
        trainer.model.eval()(
            **{k: v for k, v in model_inputs(next(iter(trainer.get_train_dataloader()))).items()}
        )
    assert not calls


# ---- option: validation, profile default, overrides ---------------------------------------------


def _parse(tmp_path, *flags, config=None, model="microsoft/phi-2"):
    import json

    argv = []
    if config is not None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        argv.append(str(path))
    return parse_args([*argv, "--model_name_or_path", model, *flags])


def test_train_fp32_tail_uses_profile_then_explicit_override(tmp_path):
    def tail(*flags, config=None):
        return _parse(tmp_path, *flags, config=config)[2].train_fp32_tail

    assert tail() == 3  # the phi profile
    assert tail(config={"train_fp32_tail": 0}) == 0
    assert tail("--train_fp32_tail", "5") == 5
    assert tail("--train_fp32_tail=0", config={"train_fp32_tail": 2}) == 0
    assert tail("--precision", "fp32") == 0  # the profile's tail belongs to mixed precision
    assert tail("--precision", "explicit", "--fp16", "--torch_dtype", "float16") == 0
    assert tail("--data_selection_method", "none") == 3  # the packed full-batch baseline too


def test_the_default_profile_and_llama_have_no_tail():
    import json

    from colm.train.config import PROFILES_FILE

    profiles = json.load(open(PROFILES_FILE))
    assert profiles["phi"]["train_fp32_tail"] == 3
    assert profiles["llama"]["train_fp32_tail"] == profiles["default"]["train_fp32_tail"] == 0


def test_train_fp32_tail_is_validated_at_load(tmp_path):
    with pytest.raises(ValueError, match="must be >= 0"):
        _parse(tmp_path, "--train_fp32_tail", "-1")
    with pytest.raises(ValueError, match="needs --fp16 or --bf16"):
        _parse(tmp_path, "--train_fp32_tail", "3", "--precision", "fp32")
    with pytest.raises(ValueError, match="needs --fp16 or --bf16"):
        make_args(
            tmp_path, train_fp32_tail=1, data_selection_method="none"
        )  # the CPU fixtures run without mixed precision
    make_args(tmp_path, train_fp32_tail=1, bf16=True, data_selection_method="none")


def test_the_coreset_selection_forward_never_sees_the_tail(
    tmp_path, tokenizer, mixture_file, monkeypatch
):
    """Selection runs another attention implementation and outside `running()`: its forwards make
    no fp32 attention call, the training forwards of the same steps do."""
    calls, seen = [], {"selection": [], "training": []}
    real = precision.fp32_attention
    monkeypatch.setattr(
        precision, "fp32_attention", lambda q, *a: (calls.append(q.dtype), real(q, *a))[1]
    )
    args = make_args(
        tmp_path,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        efficient_mezo=True,
        keep_sources="0",
        max_steps=2,
        bf16=True,
        train_fp32_tail=2,
        selection_attn_implementation="eager",
    )
    trainer, _ = build(args, tokenizer, mixture_file)

    def spy(owner, name, phase):
        original = getattr(owner, name)

        def call(*a, **k):
            before = len(calls)
            result = original(*a, **k)
            seen[phase].append(len(calls) - before)
            return result

        monkeypatch.setattr(owner, name, call)

    spy(trainer.extractor, "extract", "selection")
    spy(trainer.batching, "loss", "training")
    trainer.train()
    assert seen["selection"] and set(seen["selection"]) == {0}
    assert seen["training"] and set(seen["training"]) == {2}  # 2 tail blocks per training forward
    assert trainer.describe()["train_fp32_tail"] == 2
