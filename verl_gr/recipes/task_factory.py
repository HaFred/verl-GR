"""Task loading helpers for recipe-specific verl-GR runtime wiring."""

from __future__ import annotations

from importlib import import_module
from typing import Any

DEFAULT_TASK_CLASS_PATH = "verl_gr.recipes.openonerec.onerec_recipe.OneRecTask"


def load_object(class_path: str) -> Any:
    """Load an object by fully-qualified module path."""

    module_name, _, attr_name = class_path.rpartition(".")
    if not module_name or not attr_name:
        raise ValueError(f"Expected a fully-qualified object path, got: {class_path!r}")
    return getattr(import_module(module_name), attr_name)


def get_task_class_path(config) -> str:
    """Resolve the task class path while keeping OpenOneRec as the default."""

    task_cfg = config.get("task")
    if task_cfg is None:
        return DEFAULT_TASK_CLASS_PATH
    return str(task_cfg.get("class_path", DEFAULT_TASK_CLASS_PATH))


def build_task(config):
    """Instantiate the configured task implementation."""

    task_cls = load_object(get_task_class_path(config))
    return task_cls()
