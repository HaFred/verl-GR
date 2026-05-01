"""MiniOneRec prompt and SID formatting helpers."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np


def parse_maybe_list(value: Any) -> list[Any]:
    """Parse MiniOneRec CSV list fields without changing already parsed data."""

    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
        return parsed if isinstance(parsed, list) else [parsed]
    return [value]


def build_sid_prompt(history_item_sid: list[Any]) -> tuple[str, str]:
    """Mirror MiniOneRec.data.SidDataset prompt and history formatting."""

    history = ", ".join(str(item) for item in history_item_sid)
    history_key = "::".join(str(item) for item in history_item_sid)
    prompt = (
        "### User Input: \n"
        f"The user has interacted with items {history} in chronological order. "
        "Can you predict the next possible item that the user may expect?\n\n"
        "### Response:\n"
    )
    return prompt, history_key
