"""In-memory rotating log buffer for the dashboard log viewer."""

from __future__ import annotations

import time
from typing import Any

_BUFFER: list[dict[str, Any]] = []
_MAX = 200


def push(level: str, message: str) -> None:
    entry = {
        "timestamp": time.time(),
        "level": level,
        "message": message,
    }
    _BUFFER.append(entry)
    if len(_BUFFER) > _MAX:
        _BUFFER[:] = _BUFFER[-_MAX:]


def snapshot() -> list[dict[str, Any]]:
    return list(_BUFFER)
