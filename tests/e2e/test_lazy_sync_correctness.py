"""E2E correctness: verify lazy weight sync does not degrade training quality.

When weight_sync_interval=4 (sync every 4 steps instead of every step):
1. Training completes without errors
2. update_weights phase time is reduced (Feature 7 profiling)
3. SID accuracy does not degrade compared to sync-every-step baseline
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
def test_lazy_sync_reduces_update_weights_time():
    """Lazy sync should reduce update_weights phase time by ~75%."""
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = _run_steps(12, tmpdir, extra_config=(
            "++actor_rollout_ref.rollout.custom.weight_sync_interval=4"
        ))

        assert len(metrics) >= 12, f"Expected ≥ 12 steps, got {len(metrics)}"

        # Verify update_weights profiling metric exists
        uw_times = [
            m.get("perf/update_weights/mean", 0)
            for m in metrics if "perf/update_weights/mean" in m
        ]
        assert len(uw_times) > 0, (
            "No update_weights profiling metrics found. "
            "Feature 7 (profiling) must be active."
        )
        # With sync interval=4, amortized time should be < 0.8s (was 1.98s)
        avg_uw = sum(uw_times) / len(uw_times)
        assert avg_uw < 0.8, (
            f"update_weights time too high: {avg_uw:.2f}s. "
            f"Expected < 0.8s with interval=4 (was 1.98s baseline)."
        )


@pytest.mark.skip(reason="Requires GPU cluster with vLLM. Run on cluster with --run-gpu.")
def test_lazy_sync_no_accuracy_degradation():
    """SID accuracy should not degrade with sync-every-4 vs sync-every-step."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Baseline: sync every step
        metrics_every = _run_steps(20, tmpdir1, extra_config=(
            "++actor_rollout_ref.rollout.custom.weight_sync_interval=1"
        ))

        # Lazy sync: every 4 steps
        metrics_lazy = _run_steps(20, tmpdir2, extra_config=(
            "++actor_rollout_ref.rollout.custom.weight_sync_interval=4"
        ))

        every_score = metrics_every[-1].get("val-core/pass_at_1/mean", 0)
        lazy_score = metrics_lazy[-1].get("val-core/pass_at_1/mean", 0)

        assert lazy_score >= every_score * 0.95, (
            f"SID accuracy dropped > 5%: sync_every={every_score:.4f}, sync_lazy={lazy_score:.4f}"
        )
