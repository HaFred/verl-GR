"""SID-only log-prob scoring for OpenOneRec acceleration."""
from __future__ import annotations

import torch
from verl import DataProto


def apply_sid_only_scoring_mask(
    batch: DataProto,
    sid_token_counts: list[int],
    response_length: int,
) -> DataProto:
    """Zero out response_mask for non-SID tokens from loss computation."""
    if batch.batch is None or "response_mask" not in batch.batch:
        return batch

    response_mask = batch.batch["response_mask"]
    B = response_mask.shape[0]

    for i in range(B):
        sid_count = min(sid_token_counts[i], response_length)
        if sid_count <= 0:
            response_mask[i, :] = 0
        else:
            response_mask[i, :-sid_count] = 0

    batch.batch["response_mask"] = response_mask
    return batch
