"""Fail if library/app code uses pickle or calls torch.load without weights_only=True.

Usage: python scripts/check_no_pickle.py [ROOT ...]   (defaults: src app)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PICKLE_RE = re.compile(r"\b(c?pickle|dill|joblib)\b")
TORCH_LOAD_RE = re.compile(r"torch\.load\(([^)]*)\)", re.DOTALL)


def find_violations(roots: list[Path]) -> list[str]:
    """Return human-readable violations under the given source roots."""
    problems: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if PICKLE_RE.search(line.split("#", 1)[0]):
                    problems.append(f"{path}:{lineno}: pickle-family import/use")
            for match in TORCH_LOAD_RE.finditer(text):
                if "weights_only=True" not in match.group(1):
                    lineno = text.count("\n", 0, match.start()) + 1
                    problems.append(f"{path}:{lineno}: torch.load without weights_only=True")
    return problems


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path("src"), Path("app")]
    problems = find_violations([r for r in roots if r.exists()])
    for p in problems:
        sys.stderr.write(p + "\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
