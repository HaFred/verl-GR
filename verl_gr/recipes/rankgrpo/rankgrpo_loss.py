"""Rank-GRPO actor loss helpers."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig
from verl.workers.utils.padding import no_padding_2_padding


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _item_level_log_prob(
    *,
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    rank_seg_ids: torch.Tensor,
    response_mask: torch.Tensor,
    rec_num: int,
) -> torch.Tensor:
    """Replace token log-ratios with per-item geometric-mean log-ratios."""

    log_ratio = log_prob - old_log_prob
    seg_ids = rank_seg_ids.clamp(min=0, max=rec_num).long()
    mask = response_mask.to(dtype=log_ratio.dtype)

    batch_size, _ = seg_ids.shape
    n_bins = rec_num + 1  # final bin is overflow after the configured recommendation count
    sums = torch.zeros((batch_size, n_bins), dtype=log_ratio.dtype, device=log_ratio.device)
    counts = torch.zeros_like(sums)

    sums.scatter_add_(1, seg_ids, log_ratio * mask)
    counts.scatter_add_(1, seg_ids, mask)
    item_log_ratio = (sums / counts.clamp(min=1.0)).gather(1, seg_ids)
    return old_log_prob + item_log_ratio


def rankgrpo_ppo_loss(
    config: ActorConfig,
    rank_grpo_config,
    model_output,
    data: TensorDict,
    dp_group=None,  # noqa: ARG001
):
    """PPO loss with Rank-GRPO item-level importance sampling support."""

    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")

    importance_sampling_level = _cfg_get(rank_grpo_config, "importance_sampling_level", "token")
    if importance_sampling_level == "item":
        if "rank_seg_ids" not in data.keys():
            raise KeyError("Rank-GRPO item-level importance sampling requires `rank_seg_ids` in the batch.")
        fields.append("rank_seg_ids")

    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)
    policy_log_prob = log_prob

    metrics = {}
    if importance_sampling_level == "item":
        rec_num = int(_cfg_get(rank_grpo_config, "rec_num", 20))
        policy_log_prob = _item_level_log_prob(
            log_prob=log_prob,
            old_log_prob=old_log_prob,
            rank_seg_ids=data["rank_seg_ids"],
            response_mask=response_mask,
            rec_num=rec_num,
        )
        metrics["actor/rankgrpo_importance_sampling_item"] = Metric(value=1.0, aggregation=AggregationType.MEAN)

    loss_agg_mode = config.loss_agg_mode
    loss_mode = config.policy_loss.get("loss_mode", "vanilla")
    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=policy_log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    metrics.update(Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN))
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss -= config.entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld,
            loss_mask=response_mask,
            loss_agg_mode=config.loss_agg_mode,
            **config.global_batch_info,
        )
        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics
