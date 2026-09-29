"""Tiny random Phi + character tokenizer + synthetic data-mixture fixtures (CPU only)."""

import os
import shutil
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# Every `load_dataset("json", ...)` of a test writes an Arrow cache; keep them out of $HF_DATASETS_CACHE
# (each run added ~150 directories to the shared cache). Must be set before `datasets` is imported.
os.environ["HF_DATASETS_CACHE"] = tempfile.mkdtemp(prefix="colm-tests-datasets-")
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from equivalence import fixtures
from equivalence.fixtures import NUM_LAYERS, NUM_SOURCES, add_lora, make_phi  # noqa: F401
from equivalence.helpers import data_file


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(os.environ["HF_DATASETS_CACHE"], ignore_errors=True)


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
