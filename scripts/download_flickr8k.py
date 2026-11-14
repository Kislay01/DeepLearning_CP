"""Download, MD5-verify and extract Flickr8k (same as `captioner download`).

Usage: uv run python scripts/download_flickr8k.py [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse

from captioner.config import load_config
from captioner.data.download import download_flickr8k
from captioner.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    setup_logging()
    download_flickr8k(load_config(args.config))


if __name__ == "__main__":
    main()
