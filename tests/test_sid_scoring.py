"""Tests for SID-only log-prob scoring mask."""

import sys
from unittest.mock import MagicMock

import torch

# Mock verl.DataProto before importing the module under test
_mock_verl = MagicMock()
_mock_verl.DataProto = MagicMock()
sys.modules["verl"] = _mock_verl

# Now we can import the module (verl.DataProto is a mock)
from verl_gr.trainers.sid_scoring import apply_sid_only_scoring_mask


class _MinimalDataProto:
    """Minimal DataProto stub for testing without full verl dependency."""

    def __init__(self, batch=None, non_tensor_batch=None, meta_info=None):
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}


def test_mask_all_sid_tokens():
    """sid_count == response_length => mask unchanged (all ones remain ones)."""
    B, L = 2, 5
    mask = torch.ones(B, L)
    batch = _MinimalDataProto(batch={"response_mask": mask.clone()})

    result = apply_sid_only_scoring_mask(batch, [L, L], L)

    assert result is batch
    assert torch.equal(result.batch["response_mask"], torch.ones(B, L))


def test_mask_no_sid_tokens():
    """sid_count == 0 => all zeros."""
    B, L = 2, 5
    mask = torch.ones(B, L)
    batch = _MinimalDataProto(batch={"response_mask": mask.clone()})

    result = apply_sid_only_scoring_mask(batch, [0, 0], L)

    assert torch.equal(result.batch["response_mask"], torch.zeros(B, L))


def test_mask_partial_sid_tokens():
    """Last N tokens unmasked, rest zeroed."""
    B, L = 3, 10
    mask = torch.ones(B, L)
    batch = _MinimalDataProto(batch={"response_mask": mask.clone()})

    sid_counts = [3, 5, 2]
    result = apply_sid_only_scoring_mask(batch, sid_counts, L)

    for i in range(B):
        expected = torch.zeros(L)
        sid = sid_counts[i]
        expected[-sid:] = 1
        assert torch.equal(result.batch["response_mask"][i], expected), (
            f"Row {i}: expected {expected}, got {result.batch['response_mask'][i]}"
        )


def test_mask_sid_exceeds_response():
    """sid_count > response_length => clamp to response_length (all ones)."""
    B, L = 2, 4
    mask = torch.ones(B, L)
    batch = _MinimalDataProto(batch={"response_mask": mask.clone()})

    result = apply_sid_only_scoring_mask(batch, [10, 100], L)

    assert torch.equal(result.batch["response_mask"], torch.ones(B, L))


def test_mask_preserves_existing_values():
    """Existing zeros in mask remain zeros."""
    B, L = 1, 6
    mask = torch.ones(B, L)
    mask[0, 1] = 0  # position 1 already zero
    mask[0, 3] = 0  # position 3 already zero
    batch = _MinimalDataProto(batch={"response_mask": mask.clone()})

    result = apply_sid_only_scoring_mask(batch, [2], L)

    # Last 2 tokens (positions 4, 5) remain 1, rest zeroed
    expected = torch.zeros(B, L)
    expected[0, -2:] = 1
    assert torch.equal(result.batch["response_mask"], expected)


def test_no_response_mask_key():
    """Missing response_mask key => batch returned unchanged."""
    batch = _MinimalDataProto(batch={"some_other_key": torch.ones(3, 5)})
    result = apply_sid_only_scoring_mask(batch, [3, 3, 3], 5)
    assert result is batch
    assert "response_mask" not in result.batch


def test_none_batch():
    """None batch => batch returned unchanged."""
    batch = _MinimalDataProto(batch=None)
    result = apply_sid_only_scoring_mask(batch, [3], 5)
    assert result is batch
