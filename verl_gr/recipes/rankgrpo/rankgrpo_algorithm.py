"""Rank-GRPO advantage helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch
from verl import DataProto

from verl_gr.recipes.rankgrpo.rankgrpo_reward import rank_rewards_from_text

__all__ = ["compute_rank_grpo_advantage", "compute_rank_grpo_training_reward_metrics", "rankgrpo_enabled"]


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def rankgrpo_enabled(config: Any) -> bool:
    rank_cfg = _cfg_get(config, "rank_grpo", None)
    return bool(_cfg_get(rank_cfg, "enable", False))


def _as_float_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def compute_rank_grpo_training_reward_metrics(data: DataProto) -> dict[str, float]:
    """Expose TRL-comparable training reward scalars from Rank-GRPO rewards."""

    rank_reward_sum = data.non_tensor_batch.get("rank_reward_sum")
    rank_reward_mean = data.non_tensor_batch.get("rank_reward_mean")
    if rank_reward_sum is None or rank_reward_mean is None:
        return {}

    reward_total = _as_float_array(rank_reward_sum)
    reward = _as_float_array(rank_reward_mean)
    if reward_total.size == 0 or reward.size == 0:
        return {}

    return {
        # Mean hit count per generated recommendation list; matches TRL train/reward_total.
        "train/rankgrpo/reward_total": float(np.mean(reward_total)),
        # Mean binary hit rate over rank positions; equal to reward_total / rec_num.
        "train/rankgrpo/reward": float(np.mean(reward)),
        "train/rankgrpo/hit_any": float(np.mean(reward_total > 0.0)),
    }


def _decode_response_texts(responses: torch.Tensor, response_mask: torch.Tensor, tokenizer) -> list[str]:
    texts: list[str] = []
    for ids, mask in zip(responses, response_mask, strict=True):
        valid_ids = ids[mask.bool()].detach().cpu().tolist()
        texts.append(tokenizer.decode(valid_ids, skip_special_tokens=True))
    return texts


def _segment_rank_tokens(
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer,
    *,
    rank_separator: str,
    rec_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign natural rank ids to response tokens using newline-like separators."""

    device = responses.device
    batch_size, response_length = responses.size()
    seg_ids = torch.full((batch_size, response_length), -1, dtype=torch.long, device=device)
    try:
        separator_ids = tokenizer.encode(rank_separator, add_special_tokens=False)
    except Exception:
        separator_ids = []
    single_separator_id = int(separator_ids[0]) if len(separator_ids) == 1 else None

    for row_idx in range(batch_size):
        valid = int(response_mask[row_idx].sum().item())
        item_id = 0
        for token_idx in range(valid):
            seg_ids[row_idx, token_idx] = item_id
            token_id = int(responses[row_idx, token_idx].item())
            separator_count = 0
            if single_separator_id is not None and token_id == single_separator_id:
                separator_count = 1
            else:
                try:
                    piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False)
                except TypeError:
                    piece = tokenizer.decode([token_id])
                except Exception:
                    piece = ""
                separator_count = str(piece).count(rank_separator)
            if separator_count > 0:
                item_id += separator_count

    rank_token_mask = response_mask.bool() & (seg_ids >= 0) & (seg_ids < rec_num)
    return seg_ids, rank_token_mask


def _compute_eos_mask(responses: torch.Tensor, response_mask: torch.Tensor, tokenizer) -> torch.Tensor:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        return torch.zeros_like(response_mask, dtype=torch.bool)
    return response_mask.bool() & responses.eq(int(eos_token_id))


def compute_rank_grpo_advantage(
    data: DataProto,
    *,
    config,
    tokenizer,
    norm_adv_by_std_in_grpo: bool,
) -> DataProto:
    if tokenizer is None:
        raise ValueError("Rank-GRPO advantage computation requires the trainer tokenizer.")

    rank_cfg = _cfg_get(config, "rank_grpo", {}) or {}
    rec_num = int(_cfg_get(rank_cfg, "rec_num", 20))
    rank_separator = _cfg_get(rank_cfg, "rank_separator", "\n")
    year_tolerance = int(_cfg_get(rank_cfg, "year_tolerance", 2))
    exclude_seen = bool(_cfg_get(rank_cfg, "exclude_seen", True))
    normalize_by_std = bool(_cfg_get(rank_cfg, "normalize_by_std", norm_adv_by_std_in_grpo))
    gt_catalog_path = _cfg_get(rank_cfg, "gt_catalog_path", None)
    apply_extra_length_shaping = bool(_cfg_get(rank_cfg, "apply_extra_length_shaping", True))
    end_of_list_reward = float(_cfg_get(rank_cfg, "end_of_list_reward", 0.1))
    extra_token_penalty = float(_cfg_get(rank_cfg, "extra_token_penalty", -0.1))
    early_stop_penalty = float(_cfg_get(rank_cfg, "early_stop_penalty", -0.1))

    responses = data.batch["responses"]
    response_mask = data.batch["response_mask"]
    response_texts = _decode_response_texts(responses, response_mask, tokenizer)
    reward_models = data.non_tensor_batch.get("reward_model")
    if reward_models is None:
        raise KeyError("Rank-GRPO requires `reward_model` in data.non_tensor_batch.")

    reward_rows = [
        rank_rewards_from_text(
            text,
            reward_model,
            rec_num=rec_num,
            year_tolerance=year_tolerance,
            exclude_seen=exclude_seen,
            gt_catalog_path=gt_catalog_path,
        )
        for text, reward_model in zip(response_texts, reward_models, strict=True)
    ]
    rank_rewards = torch.tensor(reward_rows, dtype=torch.float32, device=responses.device)

    uids = data.non_tensor_batch.get("uid")
    if uids is None:
        uids = list(range(rank_rewards.size(0)))
    uid_to_indices: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid_to_indices[uid].append(idx)

    rank_advantages = torch.zeros_like(rank_rewards)
    for indices in uid_to_indices.values():
        idx_tensor = torch.tensor(indices, dtype=torch.long, device=responses.device)
        group_rewards = rank_rewards.index_select(0, idx_tensor)
        centered = group_rewards - group_rewards.mean(dim=0, keepdim=True)
        if normalize_by_std:
            std = group_rewards.std(dim=0, unbiased=False, keepdim=True)
            centered = centered / (std + 1e-4)
        rank_advantages.index_copy_(0, idx_tensor, centered)

    seg_ids, rank_token_mask = _segment_rank_tokens(
        responses,
        response_mask,
        tokenizer,
        rank_separator=rank_separator,
        rec_num=rec_num,
    )
    clamped_seg_ids = seg_ids.clamp(min=0, max=rec_num - 1)
    token_advantages = rank_advantages.gather(1, clamped_seg_ids)
    token_advantages = token_advantages * rank_token_mask.float()
    overflow_token_mask = response_mask.bool() & (seg_ids >= rec_num)

    # Zero out EOS token base advantages — reference rank_grpo_trainer L1868.
    # EOS tokens receive explicit reward/penalty via length shaping instead of
    # carrying the per-item advantage of the position they happen to land on.
    eos_mask = _compute_eos_mask(responses, response_mask, tokenizer)
    token_advantages = token_advantages * (~eos_mask).float()

    if apply_extra_length_shaping:
        valid_seg_ids = seg_ids.masked_fill(~response_mask.bool(), -1)
        items_emitted = valid_seg_ids.max(dim=1).values.add(1).clamp(min=0)

        terminated_with_eos = eos_mask.any(dim=1)
        has_overflow = overflow_token_mask.any(dim=1)
        exact_len = (items_emitted >= rec_num) & (~has_overflow) & terminated_with_eos
        early_stop = (items_emitted < rec_num) & terminated_with_eos

        token_advantages = token_advantages + extra_token_penalty * overflow_token_mask.float()
        if exact_len.any():
            token_advantages = token_advantages + exact_len.float().unsqueeze(1) * (
                end_of_list_reward * eos_mask.float()
            )
        if early_stop.any():
            token_advantages = token_advantages + early_stop.float().unsqueeze(1) * (
                early_stop_penalty * eos_mask.float()
            )

    data.batch["advantages"] = token_advantages
    data.batch["returns"] = token_advantages
    data.batch["rank_token_mask"] = rank_token_mask
    data.batch["item_token_mask"] = rank_token_mask | overflow_token_mask
    data.batch["rank_seg_ids"] = seg_ids

    # ---- Reward statistics for metrics ----
    data.non_tensor_batch["rank_reward_sum"] = np.array(rank_rewards.sum(dim=1).cpu().tolist())
    data.non_tensor_batch["rank_reward_mean"] = np.array(rank_rewards.mean(dim=1).cpu().tolist())

    return data
