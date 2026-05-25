"""E2E correctness: verify SID-only scoring does not degrade training quality.

When score_sid_only=true:
1. Training completes without errors
2. old_log_prob phase time is significantly reduced (Feature 7 profiling)
3. SID accuracy does not degrade compared to full scoring baseline
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest


def _run_steps(num_steps: int, output_dir: str, extra_config: str = "") -> list[dict[str, Any]]:
    """Run N training steps and return per-step metrics."""
    output_file = str(Path(output_dir) / "metrics.json")
    cmd = [
        "python", "-m", "verl_gr.trainers.main_ppo",
        "--config-path", "configs/verl_gr/openonerec",
        "--config-name", "grpo_trainer",
        "data.train_max_samples=16",
        "data.val_max_samples=8",
        "trainer.total_epochs=1",
        "trainer.save_freq=10000",
        "trainer.test_freq=10000",
        f"++dump_metrics_path={output_file}",
        f"trainer.total_training_steps={num_steps}",
    ]
    if extra_config:
        cmd.extend(extra_config.split())
    subprocess.run(cmd, check=True)
    with open(output_file) as f:
        return json.load(f)


@pytest.mark.skip(reason="Requires GPU cluster with vLLM. Run on cluster with --run-gpu.")
def test_sid_scoring_reduces_old_log_prob_time():
    """SID-only scoring should reduce old_log_prob phase time by 80-90%."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = _run_steps(10, tmpdir, extra_config=(
            "++actor_rollout_ref.rollout.custom.score_sid_only=true"
        ))

        assert len(metrics) >= 10, f"Expected ≥ 10 steps, got {len(metrics)}"

        # Verify old_log_prob profiling metric exists and is reasonable
        olp_times = [
            m.get("perf/old_log_prob/mean", 0)
            for m in metrics if "perf/old_log_prob/mean" in m
        ]
        assert len(olp_times) > 0, (
            "No old_log_prob profiling metrics found. "
            "Feature 7 (profiling) must be active."
        )
        # With SID-only scoring, old_log_prob should be < 1.0s (was 2.23s)
        avg_olp = sum(olp_times) / len(olp_times)
        assert avg_olp < 1.0, (
            f"old_log_prob time too high: {avg_olp:.2f}s. "
            f"Expected < 1.0s with SID-only scoring (was 2.23s baseline)."
        )


@pytest.mark.skip(reason="Requires GPU cluster with vLLM. Run on cluster with --run-gpu.")
def test_sid_scoring_no_accuracy_degradation():
    """SID accuracy should not degrade with SID-only scoring vs full scoring."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Baseline: full scoring
        metrics_full = _run_steps(10, tmpdir1, extra_config=(
            "++actor_rollout_ref.rollout.custom.score_sid_only=false"
        ))

        # SID-only scoring
        metrics_sid = _run_steps(10, tmpdir2, extra_config=(
            "++actor_rollout_ref.rollout.custom.score_sid_only=true"
        ))

        full_score = metrics_full[-1].get("val-core/pass_at_1/mean", 0)
        sid_score = metrics_sid[-1].get("val-core/pass_at_1/mean", 0)

        assert sid_score >= full_score * 0.85, (
            f"SID accuracy dropped > 15%: full={full_score:.4f}, sid_only={sid_score:.4f}"
        )
