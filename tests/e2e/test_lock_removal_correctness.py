"""E2E correctness: verify two-stage concurrent beam search completes without errors."""
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
def test_lock_removal_concurrent_beam_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.json"
        run_single_step("", str(output_file))
