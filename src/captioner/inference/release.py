"""Release artifacts: export a run to ``safetensors`` + ``vocab.json`` + ``config.yaml`` +
``model_card.md`` (FR-11), and load them back into a ``Predictor``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from captioner import __version__
from captioner.config import Config, save_config
from captioner.data.vocab import Vocabulary
from captioner.evaluation.metrics import METRIC_KEYS
from captioner.inference.predictor import Predictor
from captioner.models.captioner import build_model
from captioner.models.encoder import ResNetEncoder
from captioner.training.checkpoint import load_run
from captioner.utils.io import file_digest, read_json

RELEASE_FILES = ("model.safetensors", "vocab.json", "config.yaml", "model_card.md")
_ENCODER_PREFIX = "encoder."
_DECODER_PREFIX = "decoder."


class ReleaseError(RuntimeError):
    """A release folder is incomplete or fails verification."""


def _tensors(run_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any], Config, Vocabulary]:
    run = load_run(run_dir)
    tensors = {
        f"{_DECODER_PREFIX}{k}": v.detach().cpu().contiguous()
        for k, v in run.model.state_dict().items()
    }
    for k, v in (run.encoder_state or {}).items():
        tensors[f"{_ENCODER_PREFIX}{k}"] = v.detach().cpu().contiguous()
    return tensors, run.meta, run.cfg, run.vocab


def export_run(run_dir: Path, out_dir: Path, metrics_path: Path | None = None) -> dict[str, str]:
    """Write the release folder and return ``{file: sha256}`` (also saved as ``checksums.json``).

    Args:
        run_dir: Training run with ``best.pt``.
        out_dir: Destination, e.g. ``artifacts/release/0.1.0``.
        metrics_path: ``reports/<run>/metrics.json`` to quote in the model card (optional).
    """
    tensors, meta, cfg, vocab = _tensors(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(out_dir / "model.safetensors"),
        metadata={
            "model_kind": cfg.model.kind,
            "vocab_sha256": vocab.sha256(),
            "run": run_dir.name,
        },
    )
    vocab.save(out_dir / "vocab.json")
    save_config(cfg, out_dir / "config.yaml")
    metrics = read_json(metrics_path) if metrics_path and metrics_path.exists() else None
    (out_dir / "model_card.md").write_text(model_card(cfg, meta, vocab, metrics), encoding="utf-8")
    sums = {name: file_digest(out_dir / name) for name in RELEASE_FILES}
    (out_dir / "checksums.json").write_text(json.dumps(sums, indent=2) + "\n", encoding="utf-8")
    return sums


def _metrics_table(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return (
            "Not yet evaluated. Run `make eval RUN=<run>` and re-export to fill this table; "
            "no number here is estimated.\n"
        )
    lines = [
        "| Decoding | " + " | ".join(METRIC_KEYS) + " |",
        "|---" * (len(METRIC_KEYS) + 1) + "|",
    ]
    for beam, res in metrics["results"].items():
        vals = ["n/a" if res.get(k) is None else f"{res[k]:.4f}" for k in METRIC_KEYS]
        lines.append(f"| {beam.replace('_', ' ')} | " + " | ".join(vals) + " |")
    src = (
        f"Source: `reports/{metrics['run']}/metrics.json` "
        f"({metrics['split']} split, {metrics['n_images']} images)."
    )
    return "\n".join(lines) + f"\n\n{src}\n"


def model_card(
    cfg: Config, meta: dict[str, Any], vocab: Vocabulary, metrics: dict[str, Any] | None
) -> str:
    """Markdown model card: data, metrics, limitations, biases (FR-11)."""
    kind = "M2 soft-attention LSTM" if cfg.model.kind == "attention" else "M1 merge LSTM baseline"
    tuned = " with fine-tuned layer4" if cfg.train.finetune_encoder else ""
    bleu = meta.get("val_bleu4", float("nan"))
    exported = datetime.now(UTC).date().isoformat()
    return f"""# Model card — captioner {__version__}

- **Model:** {kind} over ResNet-50 (`{cfg.encoder.weights}`) features{tuned}.
- **Run:** `{meta.get("run_name")}` · epoch {meta.get("epoch")} · val BLEU-4 (NLTK) {bleu:.4f}
- **Code:** git `{meta.get("git_sha")}` · exported {exported}
- **Vocabulary:** {vocab.n_words} words + 4 specials (train split, min_freq {vocab.min_freq}),
  sha256 `{vocab.sha256()[:16]}…`

## Data
Flickr8k (Hodosh, Young & Hockenmaier, JAIR 2013): 8,000 Flickr photos of people and animals,
5 English captions each; official split 6,000 / 1,000 / 1,000. Images are licensed for
**non-commercial research and education only**; this model inherits that restriction.

## Metrics
{_metrics_table(metrics)}
## Intended use
Research and teaching demo of image captioning with speech output. Not a certified assistive
device: do not market or rely on it as an accessibility product without user testing.

## Limitations and biases
- Trained on 8k photos dominated by people and dogs in Western, outdoor scenes; other content is
  often described generically or wrongly.
- Captions can assume gender or age from appearance ("a man", "a girl") and can hallucinate
  objects, colours and counts.
- Training-time horizontal flips (fine-tuning only) can confuse "left"/"right".
- English only; maximum 23 words per caption.
- Not for safety-critical use.
"""


def verify_release(release_dir: Path) -> None:
    """Check that every release file exists and matches ``checksums.json``."""
    sums_path = release_dir / "checksums.json"
    if not sums_path.exists():
        raise ReleaseError(f"{sums_path} missing")
    sums = json.loads(sums_path.read_text(encoding="utf-8"))
    for name in RELEASE_FILES:
        path = release_dir / name
        if not path.exists() or file_digest(path) != sums.get(name):
            raise ReleaseError(f"{name} is missing or its SHA-256 does not match checksums.json")


def load_release(release_dir: Path, device: torch.device | None = None) -> Predictor:
    """Build a ``Predictor`` from a verified release folder (no executable formats involved)."""
    verify_release(release_dir)
    with (release_dir / "config.yaml").open(encoding="utf-8") as fh:
        cfg = Config.model_validate(yaml.safe_load(fh))
    vocab = Vocabulary.load(release_dir / "vocab.json")
    tensors = load_file(str(release_dir / "model.safetensors"))
    decoder = build_model(cfg.model, len(vocab))
    decoder.load_state_dict(
        {
            k.removeprefix(_DECODER_PREFIX): v
            for k, v in tensors.items()
            if k.startswith(_DECODER_PREFIX)
        }
    )
    encoder = ResNetEncoder(cfg.encoder.weights)
    enc_state = {
        k.removeprefix(_ENCODER_PREFIX): v
        for k, v in tensors.items()
        if k.startswith(_ENCODER_PREFIX)
    }
    if enc_state:
        encoder.load_state_dict(enc_state)
    return Predictor(decoder, vocab, cfg, encoder, device)
