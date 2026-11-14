from __future__ import annotations

import pytest

from captioner.data.text import clean_caption, detokenize, tokenize


def test_clean_caption_documented_example() -> None:
    assert clean_caption("A dog's ball - 2 times!") == "a dog s ball times"


def test_regression_D2_keeps_article_a() -> None:
    # The reference dropped every 1-character token, deleting the article "a".
    assert clean_caption("A black dog is running through the grass .") == (
        "a black dog is running through the grass"
    )
    assert tokenize(clean_caption("A man and a dog"))[0] == "a"
    assert tokenize(clean_caption("A man and a dog")).count("a") == 2


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("  Two   DOGS\tplay ", "two dogs play"),
        ("A child's toy-car.", "a child s toy car"),
        ("123 !!!", ""),
        ("café", "caf"),
    ],
)
def test_clean_caption_rules(raw: str, clean: str) -> None:
    assert clean_caption(raw) == clean


def test_detokenize() -> None:
    assert detokenize(["a", "dog", "runs"]) == "A dog runs."
    assert detokenize([]) == ""
