"""Logging setup for CLI entry points (library modules only call ``getLogger``)."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: int | str = logging.INFO) -> None:
    """Configure the root logger with a rich handler. Safe to call more than once."""
    root = logging.getLogger()
    if any(isinstance(h, RichHandler) for h in root.handlers):
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=False, show_path=False)],
    )
