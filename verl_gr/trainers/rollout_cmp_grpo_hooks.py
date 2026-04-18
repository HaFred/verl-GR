"""Trainer-side GRPO quality metrics for OpenOneRec default vs feature rollout (compare mode)."""

from __future__ import annotations

import threading
import weakref
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.reward import extract_reward

_tls = threading.local()

_VERL_COMPUTE_ADV_ORIG: Any = None
_VERL_COMPUTE_DATA_METRICS_ORIG: Any = None
_HOOKS_INSTALLED = False


def _rollout_n(trainer) -> int:
    return int(getattr(trainer.config.actor_rollout_ref.rollout, "n", 1) or 1)


def _blend_batch_with_feat_gen(batch: DataProto, feat_gen: DataProto) -> DataProto:
    """Replace generation tensors with feature rollout while keeping reward routing / uid."""
    td: dict[str, Any] = dict(batch.batch.items())
    for k in ("responses", "input_ids", "attention_mask", "position_ids", "prompts", "rollout_log_probs"):
        if k in feat_gen.batch.keys():
            td[k] = feat_gen.batch[k]
    new_batch = TensorDict(
        td,
        batch_size=(batch.batch.batch_size[0],),
        device=batch.batch.device,
    )
    return DataProto(
        batch=new_batch,
        non_tensor_batch=dict(batch.non_tensor_batch),
        meta_info=dict(batch.meta_info),
    )


def _seq_scores_from_token_level(
    token_level: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    return (token_level * response_mask).sum(dim=-1)


def _within_group_var_per_seq_scalar(
    seq_vals: torch.Tensor, rollout_n: int
) -> float:
    """Mean variance of `seq_vals` within consecutive groups of size rollout_n."""
    v = seq_vals.detach().float().cpu().view(-1)
    n = int(rollout_n)
    if n <= 1 or v.numel() % n != 0:
        return float(torch.var(v, unbiased=False).item()) if v.numel() > 1 else 0.0
    g = v.numel() // n
    vars_ = []
    for i in range(g):
        sl = v[i * n : (i + 1) * n]
        vars_.append(float(torch.var(sl, unbiased=False).item()))
    return float(np.mean(vars_)) if vars_ else 0.0


def _run_feature_grpo_quality(trainer: Any, batch: DataProto, feat_gen: DataProto) -> dict[str, float]:
    """Reward + advantage on feature completions; compare to already-computed default batch."""
    from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_response_mask

    if _VERL_COMPUTE_ADV_ORIG is None:
        return {}

    rollout_n = _rollout_n(trainer)
    adv_est = trainer.config.algorithm.adv_estimator
    norm_grpo = trainer.config.algorithm.get("norm_adv_by_std_in_grpo", True)

    mask = batch.batch["response_mask"]
    adv_d = batch.batch["advantages"]
    scores_d = batch.batch["token_level_scores"]
    seq_adv_d = _seq_scores_from_token_level(adv_d, mask)
    seq_score_d = _seq_scores_from_token_level(scores_d, mask)

    batch_f = _blend_batch_with_feat_gen(batch, feat_gen)
    for k in ("advantages", "returns", "token_level_scores", "token_level_rewards"):
        if k in batch_f.batch.keys():
            del batch_f.batch[k]

    batch_f.batch["response_mask"] = compute_response_mask(batch_f)
    reward_f, _extra_f = extract_reward(batch_f)
    batch_f.batch["token_level_scores"] = reward_f

    if trainer.config.algorithm.use_kl_in_reward:
        old_lp, _mfu = trainer._compute_old_log_prob(batch_f)
        batch_f = batch_f.union(old_lp)
        if getattr(trainer, "use_reference_policy", False):
            ref_lp = trainer._compute_ref_log_prob(batch_f)
            batch_f = batch_f.union(ref_lp)
        batch_f, _klm = apply_kl_penalty(
            batch_f, kl_ctrl=trainer.kl_ctrl_in_reward, kl_penalty=trainer.config.algorithm.kl_penalty
        )
    else:
        batch_f.batch["token_level_rewards"] = reward_f

    batch_f = _VERL_COMPUTE_ADV_ORIG(
        batch_f,
        adv_estimator=adv_est,
        gamma=trainer.config.algorithm.gamma,
        lam=trainer.config.algorithm.lam,
        num_repeat=rollout_n,
        norm_adv_by_std_in_grpo=norm_grpo,
        config=trainer.config.algorithm,
    )

    adv_f = batch_f.batch["advantages"]
    seq_adv_f = _seq_scores_from_token_level(adv_f, batch_f.batch["response_mask"])
    seq_score_f = _seq_scores_from_token_level(reward_f, batch_f.batch["response_mask"])

    metrics: dict[str, float] = {
        "openonerec_cmp_grpo/score_seq_mean_default": float(seq_score_d.mean().item()),
        "openonerec_cmp_grpo/score_seq_mean_feature": float(seq_score_f.mean().item()),
        "openonerec_cmp_grpo/score_seq_std_default": float(seq_score_d.std(unbiased=False).item())
        if seq_score_d.numel() > 1
        else 0.0,
        "openonerec_cmp_grpo/score_seq_std_feature": float(seq_score_f.std(unbiased=False).item())
        if seq_score_f.numel() > 1
        else 0.0,
        "openonerec_cmp_grpo/mean_abs_score_seq_diff": float(torch.mean(torch.abs(seq_score_d - seq_score_f)).item()),
        "openonerec_cmp_grpo/advantage_seq_mean_default": float(seq_adv_d.mean().item()),
        "openonerec_cmp_grpo/advantage_seq_mean_feature": float(seq_adv_f.mean().item()),
        "openonerec_cmp_grpo/mean_abs_advantage_seq_diff": float(torch.mean(torch.abs(seq_adv_d - seq_adv_f)).item()),
        "openonerec_cmp_grpo/within_group_adv_var_mean_default": _within_group_var_per_seq_scalar(seq_adv_d, rollout_n),
        "openonerec_cmp_grpo/within_group_adv_var_mean_feature": _within_group_var_per_seq_scalar(seq_adv_f, rollout_n),
        "openonerec_cmp_grpo/within_group_score_var_mean_default": _within_group_var_per_seq_scalar(seq_score_d, rollout_n),
        "openonerec_cmp_grpo/within_group_score_var_mean_feature": _within_group_var_per_seq_scalar(seq_score_f, rollout_n),
    }
    if seq_adv_d.numel() > 1 and seq_adv_f.numel() == seq_adv_d.numel():
        cd = seq_adv_d.detach().cpu().numpy().flatten()
        cf = seq_adv_f.detach().cpu().numpy().flatten()
        try:
            r = float(np.corrcoef(cd, cf)[0, 1])
            if np.isfinite(r):
                metrics["openonerec_cmp_grpo/adv_seq_pearson_def_feat"] = r
        except (IndexError, ValueError, FloatingPointError):
            pass
    return metrics


def _wrapped_compute_advantage(data: DataProto, *args: Any, **kwargs: Any) -> DataProto:
    feat_gen = None
    if data.meta_info is not None and "openonerec_cmp_feature_rollout" in data.meta_info:
        feat_gen = data.meta_info.pop("openonerec_cmp_feature_rollout")

    assert _VERL_COMPUTE_ADV_ORIG is not None
    out = _VERL_COMPUTE_ADV_ORIG(data, *args, **kwargs)

    tr_w = getattr(_tls, "trainer_ref", None)
    trainer = tr_w() if tr_w is not None else None
    cmp_on = bool(getattr(trainer.config.actor_rollout_ref.rollout, "compare_vanilla_vs_stage1_reuse", False))
    if trainer is not None and feat_gen is not None and cmp_on:
        try:
            extra = _run_feature_grpo_quality(trainer, out, feat_gen)
            prev = getattr(_tls, "pending_extra_metrics", None)
            if not isinstance(prev, dict):
                prev = {}
            prev.update(extra)
            _tls.pending_extra_metrics = prev
        except Exception as exc:  # pragma: no cover - diagnostic only
            import logging

            logging.getLogger(__name__).warning(
                "openonerec_cmp_grpo feature mirror failed (non-fatal): %s", exc, exc_info=True
            )
    return out


def _wrapped_compute_data_metrics(batch: DataProto, use_critic: bool = True) -> dict[str, Any]:
    assert _VERL_COMPUTE_DATA_METRICS_ORIG is not None
    m = _VERL_COMPUTE_DATA_METRICS_ORIG(batch, use_critic)
    extra = getattr(_tls, "pending_extra_metrics", None)
    if isinstance(extra, dict) and extra:
        m.update(extra)
        _tls.pending_extra_metrics = {}
    return m


def install_rollout_cmp_hooks() -> None:
    global _HOOKS_INSTALLED, _VERL_COMPUTE_ADV_ORIG, _VERL_COMPUTE_DATA_METRICS_ORIG
    if _HOOKS_INSTALLED:
        return
    import verl.trainer.ppo.ray_trainer as rt

    _VERL_COMPUTE_ADV_ORIG = rt.compute_advantage
    _VERL_COMPUTE_DATA_METRICS_ORIG = rt.compute_data_metrics
    rt.compute_advantage = _wrapped_compute_advantage
    # `compute_data_metrics` is imported into `ray_trainer`; rebind there so `fit()` picks up the wrapper.
    rt.compute_data_metrics = _wrapped_compute_data_metrics
    _HOOKS_INSTALLED = True


def set_rollout_cmp_trainer_ref(trainer: Any) -> None:
    if trainer is None:
        _tls.trainer_ref = None
    else:
        _tls.trainer_ref = weakref.ref(trainer)
