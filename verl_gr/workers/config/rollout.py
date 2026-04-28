"""Local rollout config extension for OpenOneRec-specific knobs."""

from dataclasses import dataclass

from verl.workers.config.rollout import RolloutConfig as BaseRolloutConfig


@dataclass
class RolloutConfig(BaseRolloutConfig):
    """Extends upstream rollout config with OpenOneRec compare toggle."""

    compare_vanilla_vs_stage1_reuse: bool = False
