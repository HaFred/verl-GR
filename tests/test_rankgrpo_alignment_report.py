"""Tests for RankGRPO TRL alignment report."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from verl import DataProto

from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import (
    RankGRPOAlignmentAccumulator,
    alignment_report_enabled,
    calculate_rankgrpo_logprob_gate_metrics,
    evaluate_rankgrpo_alignment_gate,
    get_rankgrpo_alignment_accumulator,
    record_rankgrpo_alignment_metrics,
    write_rankgrpo_alignment_report,
)


def test_logprob_gate_rollout_minus_ref_labels():
    b, t = 1, 4
    rollout = torch.tensor([[-0.1, -0.2, -0.3, 0.0]])
    ref = torch.tensor([[-0.12, -0.2, -0.28, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    batch = DataProto.from_single_dict(
        {
            "old_log_probs": rollout.clone(),
            "rollout_log_probs": rollout.clone(),
            "ref_log_prob": ref,
            "item_token_mask": mask,
        }
    )
    metrics = calculate_rankgrpo_logprob_gate_metrics(batch)
    assert metrics["logprob_gate/rollout_minus_rollout/abs_mean"] == 0.0
    assert metrics["logprob_gate/rollout_minus_ref/abs_mean"] > 0.0
    assert metrics["logprob_gate/bypass_mode"] == 1.0


def test_alignment_report_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_DEBUG_STEP", "10")
    acc = get_rankgrpo_alignment_accumulator()
    acc.steps.clear()
    acc.metrics_by_step.clear()

    record_rankgrpo_alignment_metrics(
        10,
        {
            "actor/kl_loss": 0.002,
            "train/rankgrpo/reward_total": 0.2,
            "train/rankgrpo/completions/mean_length": 190.0,
            "logprob_gate/rollout_minus_rollout/abs_mean": 0.0,
            "actor/debug/logprob_diff_abs": 0.02,
            "actor/pg_clipfrac": 0.0,
        },
    )

    out = write_rankgrpo_alignment_report(
        output_dir=tmp_path,
        trl_tb_dir="/nonexistent/trl_tb",
        experiment_name="test_run",
    )
    assert out is not None
    report_path, gate = out
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "RankGRPO TRL Alignment Report" in text
    assert "RUN_DEBUG_STEP" in text
    assert "Per-step alignment gate" in text
    assert "Gate Verdicts" in text
    assert "| logprob |" in text
    assert "| **combined** |" in text

    json_path = tmp_path / "logs" / "rankgrpo_align_report_step10.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["last_step"] == 10
    assert "gate" in payload


def test_offline_tb_alignment_report_writes(tmp_path):
    from verl_gr.recipes.rankgrpo.rankgrpo_logprob_metrics import write_offline_tb_alignment_report

    fork_tb = (
        "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup_verlgr/verl-gr-fork-main/"
        "tensorboard_log/RankGRPO/logprob_align_v015_TP2_g0_1"
    )
    trl_tb = (
        "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/"
        "logs/debug_precision_verlgr/runs/Jul06_12-05-38_hk01dgx028"
    )
    if not Path(fork_tb).is_dir() or not Path(trl_tb).is_dir():
        return

    out = write_offline_tb_alignment_report(
        fork_tb_dir=fork_tb,
        trl_tb_dir=trl_tb,
        output_dir=tmp_path,
        experiment_name="offline_test",
        max_step=5,
        report_stem="offline_test",
    )
    assert out is not None
    assert out.exists()
    assert "Precision Alignment" in out.read_text(encoding="utf-8")


def test_alignment_report_disabled_without_env(monkeypatch):
    monkeypatch.delenv("RUN_DEBUG_STEP", raising=False)
    assert alignment_report_enabled() is False
    assert write_rankgrpo_alignment_report(output_dir="/tmp") is None


def test_per_step_gate_kl_and_timing(monkeypatch, tmp_path):
    trl_tb = (
        "/home/dyvm6xra/dyvm6xrauser45/fred/local_backup/Rank-GRPO/"
        "logs/debug_precision_verlgr/runs/Jul06_12-05-38_hk01dgx028"
    )
    if not Path(trl_tb).is_dir():
        return

    acc = RankGRPOAlignmentAccumulator()
    acc.record(
        2,
        {
            "actor/kl_loss": 0.0002,
            "logprob_gate/rollout_minus_ref/abs_mean": 0.0057,
            "actor/debug/logprob_diff_abs": 0.0057,
            "timing_s/step": 5.0,
        },
    )
    sidecar = tmp_path / "rankgrpo_gate_sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "train/logprob_gate/rollout_minus_ref/abs_mean": {"2": 0.0057},
                "train/actor/debug/logprob_diff_abs": {"2": 0.0057},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("VERL_GR_TRL_GATE_SIDECAR", str(sidecar))
    gate = evaluate_rankgrpo_alignment_gate(acc, trl_tb_dir=trl_tb, max_step=2)
    assert gate.steps
    row = gate.steps[0]
    assert row.step == 2
    assert row.kl_ok or row.kl_rel_err is not None
    assert row.time_ok is None
    assert gate.fork_step_time_avg == 5.0
    assert gate.time_gate.passed
    assert row.logprob_ok is True
