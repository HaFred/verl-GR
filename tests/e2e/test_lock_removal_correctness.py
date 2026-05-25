"""E2E correctness: verify identical beam search outputs after lock removal."""
import json
import subprocess
import tempfile
from pathlib import Path
import pytest


def run_single_step(config_overrides: str, output_file: str) -> None:
    cmd = [
        "python", "-m", "verl_gr.trainers.main_ppo",
        "--config-path", "configs/verl_gr/openonerec",
        "--config-name", "grpo_trainer",
        "data.train_max_samples=8",
        "data.val_max_samples=8",
        "trainer.total_epochs=1",
        "trainer.save_freq=10000",
        "trainer.test_freq=10000",
        f'++dump_generated_items_path={output_file}',
    ] + config_overrides.split()
    subprocess.run(cmd, check=True)


@pytest.mark.skip(reason="Requires GPU cluster with vLLM")
def test_identical_sid_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        before_file = Path(tmpdir) / "before.json"
        after_file = Path(tmpdir) / "after.json"
        run_single_step("", str(before_file))
        run_single_step("++actor_rollout_ref.rollout.custom.no_stage2_lock=true", str(after_file))
        with open(before_file) as f:
            before_data = json.load(f)
        with open(after_file) as f:
            after_data = json.load(f)
        for (b_key, b_sids), (a_key, a_sids) in zip(sorted(before_data.items()), sorted(after_data.items())):
            assert b_key == a_key
            assert b_sids == a_sids
