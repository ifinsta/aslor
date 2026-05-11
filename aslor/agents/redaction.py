"""SecretRedactionAgent — sanitizes dicts before they are logged.

NEVER modifies the actual request payload.  Only used for log-safe copies.
"""

from __future__ import annotations

import json
from typing import Any

from aslor.logging_config import redact


def redact_for_log(obj: Any) -> Any:
    """Return a log-safe deep copy of *obj* with secrets replaced."""
    serialized = json.dumps(obj, default=str)
    sanitized = redact(serialized)
    try:
        return json.loads(sanitized)
    except json.JSONDecodeError:
        return sanitized
