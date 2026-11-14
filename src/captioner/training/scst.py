"""Self-critical sequence training (Rennie et al., CVPR 2017) with a CIDEr-D reward (ADR-026).

Teacher forcing trains on gold prefixes but the model is used on its own prefixes, and it
optimises token likelihood rather than the caption-level metric. SCST samples ``n`` captions per
image from the model, rewards each with CIDEr-D against the references, and uses the mean reward
of the *other* samples of the same image as the baseline (Luo, 2020), so no extra greedy pass is
needed. It is the single largest, best-documented improvement for captioning models.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from captioner.data.dataset import Batch
from captioner.data.vocab import END_IDX, PAD_IDX, START_IDX, UNK_IDX, Vocabulary
from captioner.evaluation.cider import CiderD
from captioner.models.captioner import CaptionModel

_BLOCKED = (PAD_IDX, START_IDX, UNK_IDX)


def _expand(state: dict[str, torch.Tensor], n: int) -> dict[str, torch.Tensor]:
    return {k: v.repeat_interleave(n, dim=0) for k, v in state.items()}


def sample_captions(
    model: CaptionModel, feats: torch.Tensor, n: int, max_len: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``n`` captions per image with gradients.

    Returns:
        ``ids`` ``(B*n, T)``, ``logp`` ``(B*n, T)`` (log-probability of each sampled token) and
        ``mask`` ``(B*n, T)`` — True up to and including ``<end>``.
    """
    rows = feats.shape[0] * n
    state = _expand(model.init_state(feats), n)
    tokens = torch.full((rows,), START_IDX, dtype=torch.long, device=feats.device)
    done = torch.zeros(rows, dtype=torch.bool, device=feats.device)
    ids, logps, masks = [], [], []
    for _ in range(max_len):
        step_logp, state, _ = model.step(tokens, state)  # (R, V)
        blocked = torch.zeros_like(step_logp, dtype=torch.bool)
        blocked[:, list(_BLOCKED)] = True
        logp = F.log_softmax(step_logp.float().masked_fill(blocked, float("-inf")), dim=-1)
        tokens = torch.multinomial(logp.detach().exp(), 1).squeeze(1)  # (R,)
        masks.append(~done)
        ids.append(tokens)
        logps.append(logp.gather(1, tokens.unsqueeze(1)).squeeze(1))
        done = done | (tokens == END_IDX)
        if bool(done.all()):
            break
    return torch.stack(ids, 1), torch.stack(logps, 1), torch.stack(masks, 1)


def leave_one_out_baseline(rewards: torch.Tensor, n: int) -> torch.Tensor:
    """For ``(B*n,)`` rewards, the mean reward of the other ``n-1`` samples of the same image."""
    grouped = rewards.view(-1, n)  # (B, n)
    others = (grouped.sum(dim=1, keepdim=True) - grouped) / (n - 1)
    return others.view(-1)


def scst_step(
    model: nn.Module,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    cider: CiderD,
    vocab: Vocabulary,
    n_samples: int,
    max_len: int,
    grad_clip: float,
) -> dict[str, float]:
    """One SCST update; returns the mean reward, advantage spread and loss."""
    optimizer.zero_grad(set_to_none=True)
    ids, logp, mask = sample_captions(model, batch.feats, n_samples, max_len)  # type: ignore[arg-type]
    images = batch.image_index.repeat_interleave(n_samples).tolist()
    words: Sequence[list[str]] = [vocab.decode(row) for row in ids.tolist()]
    rewards = torch.tensor(
        [cider.score_corpus_item(w, i) for w, i in zip(words, images, strict=True)],
        device=logp.device,
    )
    advantage = rewards - leave_one_out_baseline(rewards, n_samples)
    loss = -(advantage.unsqueeze(1) * logp * mask).sum() / mask.sum().clamp(min=1)
    loss.backward()  # type: ignore[no-untyped-call]
    nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "reward": float(rewards.mean()),
        "advantage_std": float(advantage.std()) if advantage.numel() > 1 else 0.0,
    }
