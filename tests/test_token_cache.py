"""The token-count cache gives exactly the dataset of the uncached path (CPU)."""

import copy

import numpy as np
import pytest
from equivalence.helpers import data_file

from colm.data.get_training_dataset import TokenCountCache, get_training_dataset

CONTEXT = 400


class CountingTokenizer:
    """Wraps a tokenizer and counts calls (the cache must avoid them)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def fields(data):
    return {
        name: getattr(data, name)
        for name in ("sources", "targets", "data_sources", "indices", "completion_lengths")
    } | {"mean_tokens": data.mean_tokens, "all_data_sources": data.all_data_sources}


def build(path, tokenizer, cache_dir, context=CONTEXT):
    return get_training_dataset(
        [path], tokenizer, context, token_cache_dir=cache_dir, hf_datasets_cache_dir=None
    )


def test_cached_dataset_is_identical_and_does_not_tokenise(tmp_path, tokenizer):
    path = data_file(tmp_path)
    cache = str(tmp_path / "cache")
    plain = build(path, tokenizer, None)
    counting = CountingTokenizer(tokenizer)
    first = build(path, counting, cache)  # miss: tokenises and writes
    assert counting.calls > 0 and len(list(tmp_path.glob("cache/*.npz"))) == 1
    counting.calls = 0
    second = build(path, counting, cache)  # hit: no tokeniser call
    assert counting.calls == 0
    assert fields(first) == fields(plain) == fields(second)


def test_drops_are_identical_with_the_cache(tmp_path, tokenizer):
    path = data_file(tmp_path)
    cache = str(tmp_path / "cache")
    for context in (175, 190):  # contexts that drop some examples: one file each
        expected = fields(build(path, tokenizer, None, context))
        counting = CountingTokenizer(tokenizer)
        build(path, counting, cache, context)
        counting.calls = 0
        assert fields(build(path, counting, cache, context)) == expected
        assert counting.calls == 0
    assert len(list(tmp_path.glob("cache/*.npz"))) == 2


def key(cache_dir, tokenizer, context, sources, targets):
    return TokenCountCache.for_texts(str(cache_dir), tokenizer, context, sources, targets).path


def test_the_key_follows_texts_context_and_tokenizer(tmp_path, tokenizer):
    sources, targets = ["a b", "c"], ["x</s>", "y</s>"]
    base = key(tmp_path, tokenizer, 400, sources, targets)
    assert base == key(tmp_path, tokenizer, 400, sources, targets)
    assert base != key(tmp_path, tokenizer, 401, sources, targets)
    assert base != key(tmp_path, tokenizer, 400, ["a b", "d"], targets)
    assert base != key(tmp_path, tokenizer, 400, sources, ["x</s>", "z</s>"])
    assert base != key(tmp_path, tokenizer, 400, ["a", " bc"], targets)  # boundaries count
    other = copy.deepcopy(tokenizer)
    other.backend_tokenizer.add_tokens(["new_token"])
    assert base != key(tmp_path, other, 400, sources, targets)


def test_unreadable_file_is_recomputed_and_slow_tokenizers_are_not_cached(tmp_path, tokenizer):
    path = data_file(tmp_path)
    cache = str(tmp_path / "cache")
    expected = fields(build(path, tokenizer, None))
    build(path, tokenizer, cache)
    (file,) = tmp_path.glob("cache/*.npz")
    file.write_bytes(b"not an npz")
    assert fields(build(path, tokenizer, cache)) == expected  # recomputed, rewritten
    with np.load(file) as data:
        assert data["fits"].dtype == bool
    assert TokenCountCache.for_texts(cache, object(), 400, [], []) is None
    assert TokenCountCache.for_texts("", tokenizer, 400, [], []) is None


@pytest.mark.parametrize("cache_dir", [None, ""])
def test_no_cache_dir_writes_nothing(tmp_path, tokenizer, cache_dir):
    build(data_file(tmp_path), tokenizer, cache_dir)
    assert not list(tmp_path.rglob("*.npz"))
