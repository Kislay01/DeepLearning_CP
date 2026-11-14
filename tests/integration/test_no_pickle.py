"""testing.md §3.5 — no pickle and no unsafe torch.load in src/ or app/ (regression D11)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.integration


def _checker():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "check_no_pickle", ROOT / "scripts/check_no_pickle.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_no_pickle"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_regression_D11_no_pickle_in_source() -> None:
    assert _checker().find_violations([ROOT / "src", ROOT / "app"]) == []
