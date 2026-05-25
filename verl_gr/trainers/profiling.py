"""Per-phase profiling for verl-GR training steps."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any


class StepProfiler:
    """Collects per-phase wall-clock times for each training step."""

    def __init__(self, log_every_n: int = 1):
        self.log_every_n = log_every_n
        self._phases: dict[str, list[float]] = {}
        self._step_count = 0

    def record(self, name: str, elapsed: float) -> None:
        """Record a manual phase timing (for code that can't use the context manager)."""
        if name not in self._phases:
            self._phases[name] = []
        self._phases[name].append(elapsed)

    @contextmanager
    def phase(self, name: str):
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self.record(name, elapsed)

    def step_done(self) -> dict[str, float] | None:
        self._step_count += 1
        if self._step_count % self.log_every_n != 0:
            return None

        summary = {}
        for name, times in self._phases.items():
            if times:
                summary[f"perf/{name}/mean"] = sum(times[-self.log_every_n:]) / min(self.log_every_n, len(times))
                summary[f"perf/{name}/total"] = sum(times[-self.log_every_n:])
        summary["perf/step_total"] = sum(
            summary.get(f"perf/{name}/total", 0)
            for name in self._phases
        )
        return summary
