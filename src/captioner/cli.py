"""``captioner`` command-line interface (Typer)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
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


SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", help="Override a config key, e.g. --set train.epochs=2 (repeatable)"),
]


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=x"]`` into ``{"a": {"b": 1}, "c": "x"}`` (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected key=value, got {pair!r}")
        node = out
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(raw)
    return out


def _load(config: Path, overrides: list[str] | None = None) -> Config:
    setup_logging()
    return load_config(config, overrides=parse_overrides(overrides))


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


@app.command()
def train(
    config: ConfigOpt = DEFAULT_CONFIG,
    set_: SetOpt = None,
    device: str | None = None,
) -> None:
    """Train a model (validation every epoch, early stopping on val BLEU-4)."""
    from captioner.training.train import run_training
    from captioner.utils.device import get_device

    result = run_training(_load(config, set_), get_device(device))
    console.print(
        f"[green]Run[/] {result.run_dir}\n"
        f"best val BLEU-4 {result.best_bleu4:.4f} at epoch {result.best_epoch} "
        f"({result.epochs_run} epochs run)"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
