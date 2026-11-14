"""``captioner`` command-line interface (Typer)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

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
        f"[green]Vocab[/] {out.vocab.n_words} words + 4 specials → {cfg.vocab_file}"
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


RunOpt = Annotated[
    Path | None,
    typer.Option("--run", help="Run folder or name under artifacts/runs (default: latest)"),
]


def _resolve_run(cfg: Config, run: Path | None) -> Path:
    from captioner.training.checkpoint import latest_run

    if run is None:
        return latest_run(cfg.paths.runs_dir)
    return run if run.exists() else cfg.paths.runs_dir / run


@app.command()
def evaluate(
    config: ConfigOpt = DEFAULT_CONFIG,
    run: RunOpt = None,
    split: str | None = None,
    beams: Annotated[
        list[int] | None, typer.Option("--beam", help="Beam width (repeatable)")
    ] = None,
    set_: SetOpt = None,
    device: str | None = None,
) -> None:
    """Decode a split with each beam width; write metrics.json, predictions.csv, samples.html."""
    from captioner.evaluation.evaluate import evaluate_run
    from captioner.evaluation.metrics import METRIC_KEYS
    from captioner.utils.device import get_device

    cfg = _load(config, set_)
    result = evaluate_run(cfg, _resolve_run(cfg, run), split, beams, get_device(device))
    table = Table(title=f"{result.metrics['run']} — {result.metrics['split']}")
    table.add_column("decoding")
    for key in METRIC_KEYS:
        table.add_column(key, justify="right")
    for name, res in result.metrics["results"].items():
        table.add_row(name, *["n/a" if res[k] is None else f"{res[k]:.4f}" for k in METRIC_KEYS])
    console.print(table)
    console.print(f"[green]Reports[/] → {result.report_dir}")


@app.command()
def predict(
    image: Annotated[Path, typer.Option("--image", "-i", exists=True, dir_okay=False)],
    beam: Annotated[int | None, typer.Option("--beam", "-b", min=1)] = None,
    run: RunOpt = None,
    release: Annotated[
        Path | None, typer.Option("--release", help="Exported release folder")
    ] = None,
    config: ConfigOpt = DEFAULT_CONFIG,
    device: str | None = None,
) -> None:
    """Caption one image: captioner predict --image photo.jpg --beam 3."""
    from captioner.inference.predictor import Predictor
    from captioner.inference.release import load_release
    from captioner.inference.validation import InvalidImageError, validate_upload
    from captioner.utils.device import get_device

    cfg = _load(config)
    dev = get_device(device)
    predictor = (
        load_release(release, dev)
        if release
        else Predictor.from_run(_resolve_run(cfg, run), device=dev)
    )
    try:
        validated = validate_upload(image.read_bytes(), predictor.cfg.app)
    except InvalidImageError as exc:
        console.print(f"[red]{exc.user_message}[/]")
        raise typer.Exit(code=2) from exc
    cap = predictor.caption(validated, beam=beam)
    console.print(cap.text)
    console.print(
        f"[dim]beam {cap.beam} · mean log-prob {cap.mean_logprob:.2f}"
        f"{' · low confidence' if cap.low_confidence else ''} · {cap.latency_s:.2f} s[/]"
    )


@app.command()
def export(
    config: ConfigOpt = DEFAULT_CONFIG,
    run: RunOpt = None,
    out: Annotated[Path | None, typer.Option("--out", help="Release folder")] = None,
) -> None:
    """Export a run to safetensors + vocab.json + config.yaml + model_card.md (+ checksums)."""
    from captioner import __version__
    from captioner.inference.release import export_run

    cfg = _load(config)
    run_dir = _resolve_run(cfg, run)
    out_dir = out or cfg.paths.artifacts_dir / "release" / __version__
    sums = export_run(run_dir, out_dir, cfg.paths.reports_dir / run_dir.name / "metrics.json")
    for name, digest in sums.items():
        console.print(f"{digest}  {out_dir / name}")


if __name__ == "__main__":  # pragma: no cover
    app()
