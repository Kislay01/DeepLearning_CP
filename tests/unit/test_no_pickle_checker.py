from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "check_no_pickle", ROOT / "scripts/check_no_pickle.py"
)
assert _spec and _spec.loader
checker = importlib.util.module_from_spec(_spec)
sys.modules["check_no_pickle"] = checker
_spec.loader.exec_module(checker)


def test_checker_flags_pickle_and_unsafe_torch_load(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text(
        "import pickle\nimport torch\n"
        "x = torch.load('a.pt')\ny = torch.load('b.pt', weights_only=True)\n"
    )
    problems = checker.find_violations([tmp_path])
    assert any("pickle" in p for p in problems)
    assert sum("weights_only" in p for p in problems) == 1
    assert checker.main([str(tmp_path)]) == 1


def test_repository_source_is_clean() -> None:
    assert checker.find_violations([ROOT / "src", ROOT / "app"]) == []
