"""Small I/O helpers: hashing, JSON, git SHA."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hex digest of a file, streamed in 1 MiB chunks."""
    h = hashlib.new(algorithm)
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def bytes_digest(data: bytes, algorithm: str = "sha256") -> str:
    """Hex digest of an in-memory buffer."""
    return hashlib.new(algorithm, data).hexdigest()


def read_json(path: Path) -> Any:
    """Load a UTF-8 JSON file."""
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def write_json(data: Any, path: Path) -> None:
    """Write pretty, UTF-8 JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def git_sha(cwd: Path | None = None) -> str:
    """Current git commit SHA, or ``"unknown"`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip()
