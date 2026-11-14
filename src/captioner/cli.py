"""``captioner`` command-line interface (Typer)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from captioner import __version__
from captioner.config import Config, load_config
from captioner.utils.logging import setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Image caption generator.")
console = Console()

ConfigOpt = Annotated[
    Path, typer.Option("--config", "-c", help="Experiment YAML", exists=True, dir_okay=False)
]
DEFAULT_CONFIG = Path("configs/attention.yaml")


def _load(config: Path) -> Config:
    setup_logging()
    return load_config(config)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"captioner {__version__}")


@app.command()
def download(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Download Flickr8k, verify MD5s and extract into data/raw."""
    from captioner.data.download import download_flickr8k

    layout = download_flickr8k(_load(config))
    console.print(f"[green]Data ready[/] images={layout.image_dir} text={layout.text_dir}")


@app.command()
def prepare(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Build the validated manifest and the train-only vocabulary."""
    from captioner.data.prepare import prepare_data

    cfg = _load(config)
    out = prepare_data(cfg)
    console.print(
        f"[green]Manifest[/] {len(out.manifest)} rows → {cfg.paths.manifest_path}\n"
        f"[green]Vocab[/] {out.vocab.n_words} words + 4 specials → {cfg.paths.vocab_path}"
    )


@app.command()
def extract(config: ConfigOpt = DEFAULT_CONFIG, device: str | None = None) -> None:
    """Extract frozen ResNet-50 features (pooled + spatial) for every image to HDF5."""
    from captioner.features.extract import extract_features
    from captioner.utils.device import get_device

    paths = extract_features(_load(config), get_device(device))
    for kind, path in paths.items():
        console.print(f"[green]{kind}[/] → {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
