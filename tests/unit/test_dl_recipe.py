"""ADR-025/026: CIDEr-D, label smoothing, AdamW groups, schedules, EMA, SCST."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from captioner.config import Config, ModelConfig
from captioner.data.dataset import Batch
from captioner.data.text import clean_caption, tokenize
from captioner.data.vocab import START_IDX, build_vocab
from captioner.evaluation.cider import CiderD, ngram_counts
from captioner.models.captioner import ModelOutput, build_model, caption_loss
from captioner.training.optim import Ema, Schedule, build_optimizer
from captioner.training.scst import leave_one_out_baseline, sample_captions, scst_step

GOLDEN = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "coco_golden.json").read_text()
)


def test_cider_matches_pycocoevalcap_on_same_tokens() -> None:
    from pycocoevalcap.cider.cider import Cider

    refs = [[tokenize(clean_caption(r)) for r in rs] for rs in GOLDEN["references"]]
    hyps = [tokenize(clean_caption(h)) for h in GOLDEN["hypotheses"]]
    expected, per_image = Cider().compute_score(
        {str(i): [" ".join(r) for r in rs] for i, rs in enumerate(refs)},
        {str(i): [" ".join(h)] for i, h in enumerate(hyps)},
    )
    mine = CiderD(refs)
    assert mine.corpus_score(hyps) == pytest.approx(expected, rel=1e-12)
    for i, h in enumerate(hyps):
        assert mine.score_corpus_item(h, i) == pytest.approx(per_image[i], rel=1e-12)
    assert mine.score_corpus_item(hyps[0], 0) == mine.score(hyps[0], refs[0])  # cache consistent
    assert mine.score(["zebra", "piano"], refs[0]) == 0.0
    assert mine.score(["a"], []) == 0.0
    assert CiderD(refs).corpus_score([]) == 0.0
    assert ngram_counts(["a", "b"], 2) == {("a",): 1, ("b",): 1, ("a", "b"): 1}


def test_label_smoothing() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 7)
    caps = torch.tensor([[1, 4, 5, 2], [1, 6, 2, 0]])
    lengths = torch.tensor([4, 3])
    out = ModelOutput(logits)
    plain = caption_loss(out, caps, lengths, 0.0).ce
    smooth = caption_loss(out, caps, lengths, 0.0, label_smoothing=0.1).ce
    ref = F.cross_entropy(
        logits.reshape(-1, 7), caps[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.1
    )
    torch.testing.assert_close(smooth, ref)
    assert not torch.isclose(plain, smooth)


def test_adamw_groups_exclude_bias_norm_embedding(mini_cfg: Config) -> None:
    cfg = mini_cfg.with_overrides({"train": {"optimizer": "adamw", "weight_decay": 0.01}})
    model = build_model(cfg.model, 50)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.01 and no_decay["weight_decay"] == 0.0
    no_decay_ids = {id(p) for p in no_decay["params"]}
    assert id(model.embed.weight) in no_decay_ids
    assert all(id(p) in no_decay_ids for n, p in model.named_parameters() if n.endswith("bias"))
    total = sum(len(g["params"]) for g in opt.param_groups)
    assert total == len([p for p in model.parameters() if p.requires_grad])


def _lrs(cfg: Config, steps: int) -> list[float]:
    model = nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1.0)
    sched = Schedule(opt, cfg, total_steps=steps)
    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(steps):
        sched.on_step()
        lrs.append(opt.param_groups[0]["lr"])
    return lrs


def test_cosine_schedule_with_warmup(mini_cfg: Config) -> None:
    cfg = mini_cfg.with_overrides({"train": {"scheduler": "cosine", "warmup_steps": 4}})
    lrs = _lrs(cfg, 20)
    assert lrs[0] == pytest.approx(0.25) and lrs[3] == pytest.approx(1.0)
    assert lrs[4] == pytest.approx(1.0) and lrs[-1] == pytest.approx(0.0, abs=1e-9)
    assert all(a >= b for a, b in itertools.pairwise(lrs[4:]))


def test_plateau_schedule_warmup_then_reduce(mini_cfg: Config) -> None:
    cfg = mini_cfg.with_overrides(
        {"train": {"warmup_steps": 2, "lr_patience": 0, "lr_factor": 0.5}}
    )
    model = nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1.0)
    sched = Schedule(opt, cfg, total_steps=10)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5)
    sched.on_step()
    sched.on_step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)
    sched.on_epoch(0.3)
    sched.on_epoch(0.2)  # no improvement → halve
    assert opt.param_groups[0]["lr"] == pytest.approx(0.5)
    assert sched.state_dict()["kind"] == "plateau"


def test_ema_tracks_average() -> None:
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    ema = Ema(model, decay=0.5)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema.update()  # first update copies the current weights
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update()
    assert ema.eval_model is not model
    assert ema.eval_model.weight.item() == pytest.approx(2.0)
    assert Ema(model, 0.0).eval_model is model


def test_leave_one_out_baseline() -> None:
    r = torch.tensor([1.0, 2.0, 3.0, 10.0, 10.0, 10.0])
    torch.testing.assert_close(
        leave_one_out_baseline(r, 3), torch.tensor([2.5, 2.0, 1.5, 10.0, 10.0, 10.0])
    )


class TableModel(nn.Module):
    """Learnable bigram language model ignoring the image (tests SCST in isolation)."""

    feature_kind = "pooled"

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.table = nn.Parameter(torch.zeros(vocab_size, vocab_size))

    def init_state(self, feats: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"dummy": feats.new_zeros(feats.shape[0], 1)}

    def step(
        self, prev: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], None]:
        return F.log_softmax(self.table[prev], dim=-1), state, None


def test_scst_learns_to_maximise_cider() -> None:
    torch.manual_seed(0)
    refs = [["a", "dog", "runs", "on", "grass"], ["a", "dog", "is", "running"]]
    vocab = build_vocab([*refs, ["a", "cat", "sits", "inside", "box"]], min_freq=1, max_len=10)
    cider = CiderD([refs, [["a", "cat", "sits", "inside", "box"]]])
    model = TableModel(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=0.3)
    batch = Batch(
        torch.zeros(1, 2), torch.zeros(1, 1, dtype=torch.long), torch.ones(1), torch.tensor([0])
    )
    ids, logp, mask = sample_captions(model, batch.feats, 4, 8)  # type: ignore[arg-type]
    assert ids.shape == logp.shape == mask.shape and ids.shape[0] == 4
    assert logp.requires_grad and bool(mask[:, 0].all())
    first = (
        sum(scst_step(model, batch, opt, cider, vocab, 8, 8, 5.0)["reward"] for _ in range(5)) / 5
    )
    for _ in range(150):
        scst_step(model, batch, opt, cider, vocab, 8, 8, 5.0)
    last = (
        sum(scst_step(model, batch, opt, cider, vocab, 8, 8, 5.0)["reward"] for _ in range(5)) / 5
    )
    assert last > first + 2.0, (first, last)
    best = max(range(len(vocab)), key=lambda j: model.table[START_IDX, j].item())
    assert vocab.itos[best] == "a"


def test_select_by_cider_and_scst_run(prepared_mini: Config, trained_runs: dict[str, Path]) -> None:
    from captioner.training.train import run_training

    cfg = prepared_mini.with_overrides(
        {
            "name": "recipe",
            "model": {"kind": "merge"},
            "train": {
                "epochs": 2,
                "select_metric": "cider",
                "label_smoothing": 0.1,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "warmup_steps": 2,
                "ema_decay": 0.9,
            },
        }
    )
    result = run_training(cfg)
    assert {"val/cider", "val/bleu4", "train/ce", "train/perplexity"} <= set(result.history[0])
    best_epoch = max(result.history, key=lambda h: h["val/cider"])["epoch"]
    assert result.best_epoch == int(best_epoch)
    scst = prepared_mini.with_overrides(
        {
            "name": "scst",
            "model": {"kind": "merge"},
            "train": {
                "epochs": 1,
                "objective": "scst",
                "init_from": str(trained_runs["merge"] / "best.pt"),
                "select_metric": "cider",
                "scst_samples": 3,
            },
        }
    )
    hist = run_training(scst).history
    assert "train/reward" in hist[0] and "train/ce" not in hist[0]
    assert math.isfinite(hist[0]["train/reward"])
    assert ModelConfig().kind == "attention"
