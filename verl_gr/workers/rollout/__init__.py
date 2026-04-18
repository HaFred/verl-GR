"""Rollout worker extensions for verl-GR."""

from importlib import import_module

__all__ = ["TwoStagevLLMRollout", "Stage2vLLMRollout"]


def __getattr__(name: str):
    if name in ("TwoStagevLLMRollout", "Stage2vLLMRollout"):
        return getattr(import_module("verl_gr.workers.rollout.two_stage_vllm_rollout"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
