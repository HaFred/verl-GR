"""MiniOneRec reward functions ported to verl reward API."""

from __future__ import annotations

import math
from typing import Any


def normalize_sid(text: Any) -> str:
    """Match MiniOneRec's loose completion/target stripping."""

    if text is None:
        return ""
    return str(text).split("Response:\n")[-1].strip("\n\" ")


def exact_match_reward(prediction: str, ground_truth: str) -> float:
    return 1.0 if normalize_sid(prediction) == normalize_sid(ground_truth) else 0.0


def rank_discounted_hit(prediction: str, ground_truth: str, extra_info: dict[str, Any]) -> float:
    """A per-sample approximation of MiniOneRec's rank-aware evaluation signal."""

    if exact_match_reward(prediction, ground_truth) == 0.0:
        return 0.0
    beam_index = int(extra_info.get("_beam_index", extra_info.get("beam_index", 0)) or 0)
    return 1.0 / math.log2(beam_index + 2)


def ndcg_penalties(group_size: int) -> list[float]:
    """Mirror MiniOneRec's rank-aware negative rewards."""

    raw = [-1.0 / math.log2(i + 2) for i in range(group_size)]
    denom = sum(raw)
    return [(-value / denom) for value in raw]


def is_valid_sid(prediction: str, valid_sid_set: set[str] | None = None) -> float:
    sid = normalize_sid(prediction)
    if not sid:
        return 0.0
    if valid_sid_set is None:
        return float(sid.startswith("<a_") and "<b_" in sid and "<c_" in sid)
    return float(sid in valid_sid_set)


def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Compute MiniOneRec-compatible rule reward.

    MiniOneRec's original `ndcg_rule_reward` is group-aware. verl calls the
    reward function per completion, so the scalar training reward uses exact
    match while exposing a rank-discounted hit metric for validation analysis.
    """

    extra_info = extra_info or {}
    hit = exact_match_reward(solution_str, ground_truth)
    rank_hit = rank_discounted_hit(solution_str, ground_truth, extra_info)
    valid = is_valid_sid(solution_str)
    return {
        "score": hit,
        "rule_reward": hit,
        "rank_discounted_hit": rank_hit,
        "valid_sid": valid,
        "invalid_sid": 1.0 - valid,
    }
