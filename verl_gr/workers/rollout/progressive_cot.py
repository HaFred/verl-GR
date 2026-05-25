"""Progressive CoT token budget scheduler for OpenOneRec acceleration."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ProgressiveCoTConfig:
    enabled: bool = False
    start_max_tokens: int = 1024
    end_max_tokens: int = 256
    schedule: str = "linear"
    total_steps: int = 2000


def compute_current_cot_max_tokens(step: int, config: ProgressiveCoTConfig) -> int:
    if not config.enabled or step < 0:
        return config.start_max_tokens

    progress = min(1.0, step / max(1, config.total_steps))

    if config.schedule == "cosine":
        ratio = 0.5 * (1.0 + math.cos(math.pi * progress))
    elif config.schedule == "step":
        if progress < 0.5:
            ratio = 1.0
        elif progress < 0.75:
            ratio = 0.5
        else:
            ratio = 0.0
    else:  # linear
        ratio = 1.0 - progress

    tokens = config.start_max_tokens - (1.0 - ratio) * (config.start_max_tokens - config.end_max_tokens)
    return max(config.end_max_tokens, int(tokens))
