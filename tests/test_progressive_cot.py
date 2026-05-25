"""Unit tests for progressive CoT token budget scheduler."""

import math

import pytest

from verl_gr.workers.rollout.progressive_cot import (
    ProgressiveCoTConfig,
    compute_current_cot_max_tokens,
)


class TestProgressiveCoT:

    def test_disabled_returns_start(self):
        config = ProgressiveCoTConfig(
            enabled=False,
            start_max_tokens=1024,
            end_max_tokens=256,
        )
        assert compute_current_cot_max_tokens(0, config) == 1024
        assert compute_current_cot_max_tokens(1000, config) == 1024
        assert compute_current_cot_max_tokens(2000, config) == 1024

    def test_linear_schedule(self):
        config = ProgressiveCoTConfig(
            enabled=True,
            start_max_tokens=1024,
            end_max_tokens=256,
            schedule="linear",
            total_steps=2000,
        )
        # Start: step 0 -> 1024
        assert compute_current_cot_max_tokens(0, config) == 1024
        # Middle: step 1000 -> ~640 (linear interpolation)
        middle = compute_current_cot_max_tokens(1000, config)
        assert 600 <= middle <= 680, f"Expected ~640, got {middle}"
        # End: step 2000 -> 256
        assert compute_current_cot_max_tokens(2000, config) == 256
        # Beyond end: step 3000 -> 256 (clamped)
        assert compute_current_cot_max_tokens(3000, config) == 256

    def test_cosine_schedule(self):
        config = ProgressiveCoTConfig(
            enabled=True,
            start_max_tokens=1024,
            end_max_tokens=256,
            schedule="cosine",
            total_steps=2000,
        )
        # Start: step 0 -> 1024
        assert compute_current_cot_max_tokens(0, config) == 1024
        # Middle: step 1000 -> ~640 (cosine at π/2 gives 0.5)
        middle = compute_current_cot_max_tokens(1000, config)
        assert 600 <= middle <= 680, f"Expected ~640, got {middle}"
        # End: step 2000 -> 256
        assert compute_current_cot_max_tokens(2000, config) == 256

    def test_step_schedule(self):
        config = ProgressiveCoTConfig(
            enabled=True,
            start_max_tokens=1024,
            end_max_tokens=256,
            schedule="step",
            total_steps=2000,
        )
        # Phase 1: progress < 0.5 -> ratio = 1.0 -> 1024
        assert compute_current_cot_max_tokens(0, config) == 1024
        assert compute_current_cot_max_tokens(500, config) == 1024
        # Edge at progress = 0.5 (step=1000): still <0.5? No, 1000/2000 = 0.5, which is not < 0.5
        # So at step 1000, progress = 0.5, ratio = 0.5 -> tokens = 1024 - 0.5*(1024-256) = 640
        step1000 = compute_current_cot_max_tokens(1000, config)
        assert 600 <= step1000 <= 680, f"Expected ~640, got {step1000}"
        # Phase 2: 0.5 <= progress < 0.75 -> ratio = 0.5 -> 640
        step1250 = compute_current_cot_max_tokens(1250, config)
        assert 600 <= step1250 <= 680, f"Expected ~640, got {step1250}"
        # Phase 3: progress >= 0.75 -> ratio = 0.0 -> 256
        assert compute_current_cot_max_tokens(1500, config) == 256
        assert compute_current_cot_max_tokens(2000, config) == 256

    def test_never_below_end(self):
        config = ProgressiveCoTConfig(
            enabled=True,
            start_max_tokens=1024,
            end_max_tokens=256,
            schedule="linear",
            total_steps=2000,
        )
        # Well past the end, should clamp to end_max_tokens
        assert compute_current_cot_max_tokens(10000, config) == 256
        # With cosine, same
        config.schedule = "cosine"
        assert compute_current_cot_max_tokens(10000, config) == 256
        # With step, same
        config.schedule = "step"
        assert compute_current_cot_max_tokens(10000, config) == 256

    def test_negative_step(self):
        config = ProgressiveCoTConfig(
            enabled=True,
            start_max_tokens=1024,
            end_max_tokens=256,
            schedule="linear",
            total_steps=2000,
        )
        # Negative step should return start_max_tokens
        assert compute_current_cot_max_tokens(-1, config) == 1024
        assert compute_current_cot_max_tokens(-100, config) == 1024

    def test_zero_total_steps(self):
        config = ProgressiveCoTConfig(
            enabled=True,
            start_max_tokens=1024,
            end_max_tokens=256,
            schedule="linear",
            total_steps=0,
        )
        # total_steps=0 -> progress clamps to 1.0 for any positive step
        # step=0 -> progress = min(1.0, 0/1) = 0.0 -> tokens = 1024
        assert compute_current_cot_max_tokens(0, config) == 1024
        # step=1 -> progress = min(1.0, 1/1) = 1.0 -> ratio = 0.0 -> tokens = 256
        assert compute_current_cot_max_tokens(1, config) == 256

    def test_custom_start_end(self):
        config = ProgressiveCoTConfig(
            enabled=True,
            start_max_tokens=512,
            end_max_tokens=128,
            schedule="linear",
            total_steps=1000,
        )
        assert compute_current_cot_max_tokens(0, config) == 512
        assert compute_current_cot_max_tokens(1000, config) == 128
        # Middle: ~320
        middle = compute_current_cot_max_tokens(500, config)
        assert 300 <= middle <= 340, f"Expected ~320, got {middle}"
