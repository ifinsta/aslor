"""ModelDetectorAgent — determines provider, model capability, and session key."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from aslor.config import AppConfig
from aslor.models.registry import get_capability


@dataclass
class DetectionResult:
    provider: str
    model_name: str
    needs_repair: bool
    session_key: str
    reasoning_field: str


def detect(body: dict[str, Any], config: AppConfig, *, session_hint: str | None = None) -> DetectionResult:
    """Analyse the request body and return a :class:`DetectionResult`."""
    model_name = body.get("model", "")
    capability = get_capability(model_name)
    provider = config.provider.name

    inferred_field = _infer_reasoning_field(body, model_name)
    needs_repair = capability.reasoning or bool(inferred_field)

    session_key = _derive_session_key(body, session_hint=session_hint)

    return DetectionResult(
        provider=provider,
        model_name=model_name,
        needs_repair=needs_repair,
        session_key=session_key,
        reasoning_field=capability.reasoning_field or inferred_field,
    )


def _derive_session_key(body: dict[str, Any], *, session_hint: str | None = None) -> str:
    """Derive a stable session key from the conversation fingerprint.

    The key is based on: lowercased model name + first-user-message content.
    Model name is lowercased so that ``deepseek-reasoner`` and
    ``DeepSeek-Reasoner`` produce the same key.  The first user message is
    trimmed to 512 characters to keep the input bounded.
    """
    model = body.get("model", "").lower()
    if isinstance(session_hint, str) and session_hint.strip():
        raw = f"{model}:{session_hint.strip()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]
    stable = _stable_conversation_id(body)
    if stable:
        raw = f"{model}:{stable}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    messages = body.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    first_user = ""
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "user":
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, sort_keys=True)
            first_user = content.strip()
            break
    raw = f"{model}:{first_user[:512]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _stable_conversation_id(body: dict[str, Any]) -> str:
    candidates: list[Any] = [
        body.get("conversation_id"),
        body.get("thread_id"),
        body.get("session_id"),
        body.get("chat_id"),
    ]
    metadata = body.get("metadata")
    if isinstance(metadata, dict):
        candidates.extend(
            [
                metadata.get("conversation_id"),
                metadata.get("thread_id"),
                metadata.get("session_id"),
                metadata.get("chat_id"),
                metadata.get("id"),
            ]
        )
    candidates.append(body.get("user"))

    for c in candidates:
        if c is None:
            continue
        if isinstance(c, str):
            s = c.strip()
        else:
            try:
                s = json.dumps(c, sort_keys=True, separators=(",", ":"))
            except Exception:
                continue
        if not s or s.lower() in {"null", "none", "unknown"}:
            continue
        if len(s) < 6:
            continue
        return s
    return ""


def _infer_reasoning_field(body: dict[str, Any], model_name: str) -> str:
    messages = body.get("messages", [])
    if isinstance(messages, list):
        for m in messages:
            if not isinstance(m, dict):
                continue
            if "thinking" in m:
                return "thinking"
            if "reasoning_content" in m:
                return "reasoning_content"

    if "reasoning_effort" in body or "reasoning" in body:
        return "reasoning_content"

    lower = str(model_name or "").lower()
    if re.search(r"(^|[^a-z0-9])o(1|3|4)([^a-z0-9]|$)", lower):
        return "reasoning_content"
    if "deepseek" in lower and ("reasoner" in lower or "r1" in lower):
        return "reasoning_content"

    return ""
