"""Structured JSON logging with secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any


_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9]{20,})", re.IGNORECASE),
    re.compile(r"(pk-[A-Za-z0-9]{20,})", re.IGNORECASE),
    re.compile(r"(ghp_[A-Za-z0-9]{20,})", re.IGNORECASE),
    re.compile(r"(Bearer\s+)([A-Za-z0-9\-._~+/]{20,})", re.IGNORECASE),
    re.compile(r"(api[_-]?key\s*[:=]\s*)([A-Za-z0-9\-._~+/]{10,})", re.IGNORECASE),
]


def redact(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        if pattern.groups == 1:
            text = pattern.sub("[REDACTED]", text)
        else:
            text = pattern.sub(lambda m: m.group(1) + "[REDACTED]", text)
    return text


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    root.handlers.clear()
    root.addHandler(handler)
