"""Backward-compatible import shim for rollout registration helpers."""

from __future__ import annotations

from verl_gr.workers.rollout.registration import (  # noqa: F401
    register_constrained_beam_replica,
    register_constrained_beam_rollout_class,
    register_two_stage_replica,
    register_two_stage_rollout_class,
)
