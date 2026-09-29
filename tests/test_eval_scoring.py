"""Accuracy scoring of math_eval: what a perfect model would score, and what counts as correct."""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scoring():
    sys.path.insert(0, str(REPO / "math_eval"))
    import data_loader
    import utils

    yield data_loader, utils
    sys.path.remove(str(REPO / "math_eval"))


def test_normalize_answer(scoring):
    _, utils = scoring
    assert utils.normalize_answer("1363.0") == "1363"
    assert utils.normalize_answer(0.0) == "0"
    assert utils.normalize_answer("2.50") == "2.5"
    assert utils.normalize_answer("[3]") == "3"
    assert utils.normalize_answer("B") == "B"
    assert utils.normalize_answer("-20*x") == "-20*x"


@pytest.mark.parametrize(
    ("dataset", "unreachable"),
    [("gsm8k", 0), ("svamp", 0), ("simuleq", 0), ("numglue", 0.005), ("deepmind", 0.05)],
)
def test_a_perfect_model_scores_almost_100(scoring, dataset, unreachable):
    """Feeding the ground truth back through the answer extraction must give the ground truth."""
    data_loader, utils = scoring
    _, answers = data_loader.data_reader(dataset)
    misses = [
        a
        for a in answers
        if utils.answer_clean(dataset, ("####", "The answer is"), f"The answer is {a}") != a
    ]
    # What is left: several answers in one, algebraic expressions and words.
    assert len(misses) <= unreachable * len(answers), misses[:20]


def test_numeric_comparison_is_not_lenient(scoring):
    _, utils = scoring
    assert utils.compare_two_numbers(3.0, 3)
    assert not utils.compare_two_numbers(2.6, 3)  # round(2.6) == 3 counted as correct
    assert not utils.compare_two_numbers(3.1, 3.0)  # 4% tolerance counted as correct
    assert utils.compare_two_numbers(0.1 + 0.2, 0.3)


def test_the_article_a_is_not_an_option(scoring):
    _, utils = scoring
    trigger = ("####", "The answer is")
    assert utils.answer_clean("numglue", trigger, "The answer is a large 5") == "5"
    assert utils.answer_clean("numglue", trigger, "The answer is B") == "B"
    assert utils.answer_clean("gsm8k", trigger, "The answer is .") == ""  # used to raise IndexError


def test_stop_strings_are_prompt_markers_not_words():
    sys.path.insert(0, str(REPO / "math_eval"))
    try:
        import run_open

        assert all(":" in s or s.startswith("###") for s in run_open.STOP_TOKENS)
    finally:
        sys.path.remove(str(REPO / "math_eval"))
