"""Loading and comparing golden files (see generate_golden.py)."""

import os

import torch

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")


def load(name: str):
    return torch.load(os.path.join(GOLDEN_DIR, f"{name}.pt"), weights_only=False)


def assert_same(a, b, path="", atol=0.0):
    """Recursive equality of nested dict / list / tensor / scalar structures."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        a, b = torch.as_tensor(a), torch.as_tensor(b)
        assert a.shape == b.shape, f"{path}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        if atol == 0.0:
            assert torch.equal(a.to(b.dtype), b), f"{path}: tensors differ"
        else:
            torch.testing.assert_close(a.double(), b.double(), rtol=0, atol=atol, msg=lambda m: f"{path}: {m}")
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{path}: keys {sorted(a)} != {sorted(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{path}/{k}", atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            assert_same(x, y, f"{path}[{i}]", atol)
    else:
        assert a == b or (a != a and b != b), f"{path}: {a!r} != {b!r}"
