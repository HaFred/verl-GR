"""Rank-GRPO task runtime wiring."""

from __future__ import annotations

from typing import Any

from omegaconf import open_dict

from verl_gr.recipes.rankgrpo.rankgrpo_worker import RankGRPOActorRolloutRefWorker
from verl_gr.recipes.task_runtime import RecipeTaskRuntime

__all__ = ["RankGRPOTask"]


class RankGRPOTask(RecipeTaskRuntime):
    """Rank-GRPO task-specific runtime preparation."""

    def prepare(self, config) -> dict[str, Any]:
        with open_dict(config.actor_rollout_ref):
            config.actor_rollout_ref.rank_grpo = config.algorithm.get("rank_grpo", {}) or {}

        actor_strategy = self._ensure_role_strategy(config, "actor")
        if actor_strategy == "ddp":
            # Side-effect: register DDP engine with verl's EngineRegistry
            import verl_gr.workers.engine.ddp  # noqa: F401

        return super().prepare(config)

    def get_actor_rollout_ref_worker(self, config):
        return RankGRPOActorRolloutRefWorker
