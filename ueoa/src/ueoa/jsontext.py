"""Pull one JSON object out of model or agent text (plain JSON or a fenced ```json block)."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    candidates = [text.strip()]
    candidates += _FENCE.findall(text)
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value  # pyright: ignore[reportUnknownVariableType]
    return None
