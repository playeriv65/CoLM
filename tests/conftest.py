"""Tiny random Phi + character tokenizer + synthetic data-mixture fixtures (CPU only)."""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from equivalence import fixtures
from equivalence.fixtures import NUM_LAYERS, NUM_SOURCES, add_lora, make_phi  # noqa: F401
from equivalence.helpers import data_file


@pytest.fixture(scope="session")
def tokenizer():
    return fixtures.tokenizer()


@pytest.fixture()
def mixture_file(tmp_path):
    """jsonl in the MathInstruct format: instruction/input/output/source."""
    return data_file(tmp_path)


@pytest.fixture()
def phi(tokenizer):
    return make_phi(tokenizer)
