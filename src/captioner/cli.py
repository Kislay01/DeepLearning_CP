"""``captioner`` command-line interface (Typer)."""

from __future__ import annotations

import typer
from rich.console import Console

from captioner import __version__

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Image caption generator.")
console = Console()


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"captioner {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
