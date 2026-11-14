"""testing.md §3.2 — overfit one batch: the main bug detector for the DL code (ADR-007)."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from captioner.config import Config, ModelConfig
from captioner.data.dataset import Batch, CaptionTable, build_caption_table, open_feature_store
from captioner.data.prepare import load_prepared
from captioner.features.extract import feature_paths
from captioner.inference.decode import greedy_decode
from captioner.models.captioner import build_model, feature_kind_for
from captioner.training.loop import Amp, train_step
from captioner.utils.seed import seed_everything

pytestmark = pytest.mark.integration


def _one_batch(cfg: Config, kind: str, n: int = 8) -> tuple[Batch, CaptionTable]:
    prep = load_prepared(cfg)
    table = build_caption_table(prep.manifest, "train", prep.vocab)
    mcfg = ModelConfig(kind=kind)  # type: ignore[arg-type]
    store = open_feature_store(feature_paths(cfg)[feature_kind_for(mcfg)], table)
    feats = torch.stack([store.get(int(table.feature_idx[i])) for i in range(n)])
    # Random-init ResNet activations are ~40x larger than ImageNet-pretrained ones (no weights
    # in CI, ADR-009); rescale to a pretrained-like range so the test exercises the decoder.
    feats = feats / feats.std()
    lengths = torch.as_tensor(table.lengths[:n, 0])
    captions = torch.as_tensor(table.captions[:n, 0, : int(lengths.max())])
    return Batch(feats, captions, lengths, torch.arange(n)), table


@pytest.mark.parametrize("kind", ["merge", "attention", "transformer"])
def test_overfit_one_batch(prepared_mini: Config, kind: str) -> None:
    seed_everything(0)
    batch, _ = _one_batch(prepared_mini, kind)
    mcfg = ModelConfig.model_validate(
        {
            "kind": kind, "embed_dim": 128, "decoder_dim": 256 if kind != "transformer" else 128,
            "attention_dim": 64, "dropout": 0.0, "transformer_layers": 2, "transformer_heads": 4,
            "transformer_ff": 256,
        }
    )  # fmt: skip
    vocab = load_prepared(prepared_mini).vocab
    model = build_model(mcfg, len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3 if kind != "transformer" else 1e-3)
    amp = Amp(torch.device("cpu"), enabled=False)
    alpha_c = 1.0 if kind == "attention" else 0.0
    ce, reached = float("inf"), None
    for step in range(1, 301):
        ce = float(train_step(model, batch, opt, amp, grad_clip=5.0, alpha_c=alpha_c).ce)
        if ce < 0.5 and reached is None:
            reached = step
        if ce < 0.02:
            break
    assert reached is not None, f"{kind}: cross-entropy {ce:.3f} after 300 steps"
    model.eval()
    hyps = greedy_decode(model, batch.feats, max_len=25)
    exact = [
        list(h.ids) == batch.captions[i, 1 : int(batch.lengths[i]) - 1].tolist()
        for i, h in enumerate(hyps)
    ]
    assert any(exact), f"{kind}: no training caption reproduced exactly"
    assert dataclasses.is_dataclass(hyps[0])
