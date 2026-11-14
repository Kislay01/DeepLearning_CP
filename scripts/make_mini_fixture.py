"""Build tests/fixtures/mini_flickr from the full extracted Flickr8k (ADR-002).

Picks 12 train / 4 val / 4 test images deterministically, re-encodes them (shorter side 256 px,
JPEG q=70) so the whole fixture stays < 1 MB, and writes the matching subset of the token file
(plus the orphan id's captions, to keep the D8 quirk) and split lists.

Usage: uv run python scripts/make_mini_fixture.py [--config configs/base.yaml] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from captioner.config import load_config
from captioner.data.download import raw_layout
from captioner.data.manifest import read_split_file

COUNTS = {"train": 12, "val": 4, "test": 4}
ORPHAN = "2258277193_586949ec62.jpg.1"
SHORT_SIDE = 256
QUALITY = 70


def pick(names: list[str], k: int) -> list[str]:
    """Evenly spaced picks from the sorted list (deterministic)."""
    ordered = sorted(names)
    step = len(ordered) / k
    return [ordered[int(i * step)] for i in range(k)]


def reencode(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        rgb = img.convert("RGB")
        scale = SHORT_SIDE / min(rgb.size)
        size = (round(rgb.width * scale), round(rgb.height * scale))
        rgb.resize(size, Image.Resampling.BICUBIC).save(dst, "JPEG", quality=QUALITY, optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--out", default="tests/fixtures/mini_flickr")
    args = parser.parse_args()
    cfg = load_config(args.config)
    layout = raw_layout(cfg)
    out = Path(args.out)
    img_out = out / cfg.dataset.image_dir_name
    txt_out = out / cfg.dataset.text_dir_name
    img_out.mkdir(parents=True, exist_ok=True)
    txt_out.mkdir(parents=True, exist_ok=True)

    chosen: dict[str, list[str]] = {}
    for split, k in COUNTS.items():
        fname = cfg.dataset.split_files[split]
        chosen[split] = pick(read_split_file(layout.text_dir / fname), k)
        (txt_out / fname).write_text("\n".join(chosen[split]) + "\n", encoding="utf-8")
    keep = {n for names in chosen.values() for n in names} | {ORPHAN}

    lines = (layout.text_dir / cfg.dataset.token_file).read_text(encoding="utf-8").splitlines()
    subset = [ln for ln in lines if ln.split("\t", 1)[0].rsplit("#", 1)[0] in keep]
    (txt_out / cfg.dataset.token_file).write_text("\n".join(subset) + "\n", encoding="utf-8")

    for names in chosen.values():
        for name in names:
            reencode(layout.image_dir / name, img_out / name)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {len(keep) - 1} images, {len(subset)} caption lines, {total / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
