"""Numerical checks for RankGRPO `trl_match` vs dual-clip `verl_default` loss behavior."""

from __future__ import annotations

import sys
from pathlib import Path

# `python tests/foo.py` puts `tests/` first on sys.path, not the repo root — ensure root
# (the directory that contains `verl_gr/`) is on sys.path.
_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "verl_gr").is_dir():
    _p = _p.parent
if (_p / "verl_gr").is_dir() and str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import torch

from verl.trainer.ppo.core_algos import agg_loss

from verl_gr.recipes.rankgrpo.rankgrpo_loss import (
    _compute_item_mean_log_ratio,
    _trl_clipped_pg_loss,
)


def _dual_clip_pg_losses(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    clip_ratio_c: float,
) -> torch.Tensor:
    """Mirror of `compute_policy_loss_vanilla` surrogate (per-token, before agg)."""

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    return torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)


def test_trl_clipped_pg_matches_manual_min_formulation():
    torch.manual_seed(0)
    b, t = 3, 7
    w = torch.randn(b, t, dtype=torch.float64) * 0.15
    adv = torch.randn(b, t, dtype=torch.float64) * 0.4
    eps_l, eps_h = 0.2, 0.2

    out = _trl_clipped_pg_loss(
        log_importance_weights=w,
        advantages=adv,
        loss_mask=torch.ones(b, t, dtype=torch.bool),
        clip_ratio_low=eps_l,
        clip_ratio_high=eps_h,
        kl_per_token=None,
        kl_coef=0.0,
    )
    c1 = torch.exp(w)
    c2 = torch.clamp(c1, 1 - eps_l, 1 + eps_h)
    expected = -torch.min(c1 * adv, c2 * adv)
    assert torch.allclose(out, expected, rtol=0, atol=1e-12)


def test_item_mean_log_ratio_broadcasts_within_segments():
    """Two sequences, two items each: mean log-ratio per item then gather to tokens."""

    b, t, rec_num = 2, 4, 2
    # seg 0 and 1 per row
    rank_seg_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 0]], dtype=torch.long)
    log_prob = torch.tensor(
        [[0.0, 0.0, 1.0, -1.0], [0.5, 0.5, 0.5, 0.5]], dtype=torch.float64
    )
    old_log_prob = torch.zeros(b, t, dtype=torch.float64)
    mask = torch.ones(b, t, dtype=torch.bool)

    liw = _compute_item_mean_log_ratio(
        log_prob=log_prob,
        old_log_prob=old_log_prob,
        rank_seg_ids=rank_seg_ids,
        response_mask=mask,
        rec_num=rec_num,
    )
    # row0 item0 mean = 0, item1 mean = 0
    assert torch.allclose(liw[0, :2], torch.zeros(2, dtype=torch.float64))
    assert torch.allclose(liw[0, 2:], torch.zeros(2, dtype=torch.float64))
    # row1 item0 tokens at 0,3: log ratios 0.5,0.5 -> 0.5; item1 at 1,2: 0.5
    assert torch.allclose(liw[1], torch.full((t,), 0.5, dtype=torch.float64))


def test_trl_match_agg_differs_from_dual_clip_when_negative_adv_and_large_ratio():
    """When adv < 0 and ratio is above the upper clip, dual-clip caps loss at -adv * clip_ratio_c."""

    b, t = 1, 4
    old_lp = torch.zeros(b, t, dtype=torch.float64)
    # Large positive log_ratio -> ratio > 1+eps so PPO clip engages; dual-clip then min(..., -adv*3).
    log_prob = torch.full((b, t), 2.0, dtype=torch.float64)
    advantages = torch.full((b, t), -1.0, dtype=torch.float64)
    loss_mask = torch.ones(b, t, dtype=torch.bool)
    eps_l, eps_h = 0.2, 0.2
    clip_c = 3.0

    trl_tok = _trl_clipped_pg_loss(
        log_importance_weights=log_prob - old_lp,
        advantages=advantages,
        loss_mask=loss_mask,
        clip_ratio_low=eps_l,
        clip_ratio_high=eps_h,
        kl_per_token=None,
        kl_coef=0.0,
    )
    dual_tok = _dual_clip_pg_losses(log_prob, old_lp, advantages, eps_l, eps_h, clip_c)

    global_info = dict(dp_size=1, batch_num_tokens=None, global_batch_size=None, loss_scale_factor=None)
    trl_scalar = agg_loss(trl_tok, loss_mask, "seq-mean-token-mean", **global_info)
    dual_scalar = agg_loss(dual_tok, loss_mask, "seq-mean-token-mean", **global_info)
    assert not torch.allclose(trl_scalar, dual_scalar, rtol=1e-2, atol=1e-6), (
        "expected dual-clip and trl_match to differ on this synthetic batch"
    )


if __name__ == "__main__":
    test_trl_clipped_pg_matches_manual_min_formulation()
    test_item_mean_log_ratio_broadcasts_within_segments()
    test_trl_match_agg_differs_from_dual_clip_when_negative_adv_and_large_ratio()
    print("test_rankgrpo_loss_modes: all checks passed")
