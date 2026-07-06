"""Rank-GRPO worker customizations."""

from __future__ import annotations

from functools import partial

from omegaconf import OmegaConf
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import ActorConfig
from verl.workers.engine_workers import ActorRolloutRefWorker

from verl_gr.recipes.rankgrpo.rankgrpo_loss import rankgrpo_ppo_loss
from verl_gr.workers.grad_hooks import install_grad_hooks


class RankGRPOActorRolloutRefWorker(ActorRolloutRefWorker):
    """Actor/rollout/ref worker that installs the Rank-GRPO PPO loss."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        install_grad_hooks()

        if not (self._is_actor and self.actor is not None and not self.distillation_enabled):
            return

        actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
        rank_grpo_config = self.config.get("rank_grpo", {}) or {}
        if OmegaConf.is_config(rank_grpo_config):
            rank_grpo_config = OmegaConf.to_container(rank_grpo_config, resolve=True)

        self.loss_fn = partial(rankgrpo_ppo_loss, config=actor_config, rank_grpo_config=rank_grpo_config)
        self.actor.set_loss_fn(self.loss_fn)

        for role_worker in (getattr(self, "actor", None), getattr(self, "ref", None)):
            if role_worker is not None and hasattr(role_worker, "engine_config"):
                setattr(role_worker.engine_config, "completion_only_logprob", True)
                role_name = "actor" if role_worker is getattr(self, "actor", None) else "ref"
                flag = getattr(role_worker.engine_config, "completion_only_logprob", False)
                print(f"[rankgrpo] {role_name} completion_only_logprob={flag}", flush=True)
