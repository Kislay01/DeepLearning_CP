from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from captioner.evaluation.taxonomy import CATEGORIES, classify, diversity, taxonomy_counts
from captioner.inference.heatmap import (
    attention_grid,
    attention_summary,
    model_view,
    overlay,
    region_name,
)

REFS = ["A black dog runs on the grass.", "A dog is running outside.", "Dog in a field"]


@pytest.mark.parametrize(
    ("pred", "expected"),
    [
        ("a black dog runs on the grass", set()),
        ("a white dog runs on the grass", {"wrong_colour"}),
        ("two dogs run on the grass", {"wrong_count"}),
        ("a man runs on the grass", {"wrong_object", "gender_age_assumption"}),
        ("a dog runs with a ball", {"hallucinated_object"}),
        (
            "a man is standing",
            {"wrong_object", "gender_age_assumption", "hallucinated_object", "generic_caption"},
        ),
        ("", {"generic_caption"}),
    ],
)
def test_classify(pred: str, expected: set[str]) -> None:
    assert classify(pred, REFS) == expected


def test_counts_and_diversity() -> None:
    counts = taxonomy_counts(["a white dog runs", "a dog runs"], [REFS, REFS])
    assert set(counts) == set(CATEGORIES) and counts["wrong_colour"] == 1
    div = diversity(["a dog runs", "a dog runs", "a cat sits"])
    assert div["unique_caption_ratio"] == pytest.approx(2 / 3)
    assert div["vocab_used"] == 5 and div["mean_length"] == 3


def test_heatmap_helpers() -> None:
    grid = np.zeros((7, 7), dtype=np.float32)
    grid[6, 0] = 1.0
    assert region_name(grid) == "lower-left"
    grid2 = np.zeros((7, 7), dtype=np.float32)
    grid2[3, 3] = 1.0
    assert region_name(grid2) == "center"
    grid3 = np.zeros((7, 7), dtype=np.float32)
    grid3[3, 6] = 1.0
    assert region_name(grid3) == "middle-right"
    assert attention_grid(np.arange(49)).shape == (7, 7)
    img = Image.new("RGB", (400, 300), (10, 200, 30))
    assert model_view(img).size == (224, 224)
    assert overlay(img, grid).size == (224, 224)
    assert overlay(img, np.zeros((7, 7), dtype=np.float32)).size == (224, 224)
    assert (
        attention_summary(["dog", "grass"], [grid, grid2]) == "'dog' → lower-left; 'grass' → center"
    )
