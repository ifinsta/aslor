"""ResponseCaptureAgent - extracts and caches reasoning state from responses.

For non-streaming responses the full body is available immediately.
For streaming responses, the caller (StreamRelayAgent) accumulates the chunks
and calls ``capture_from_assembled`` once the stream is complete.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from aslor.providers.base import ProviderAdapter
from aslor.reasoning.state import ReasoningStateStore

logger = logging.getLogger(__name__)


def capture_from_assembled(
    assembled_response: dict[str, Any],
    session_id: str,
    adapter: ProviderAdapter,
    store: ReasoningStateStore,
) -> None:
    """Extract reasoning state from *assembled_response* and persist it."""
    state = adapter.extract_reasoning_state(assembled_response)
    if state:
        message = _extract_assistant_message(assembled_response)
        if message is not None:
            store.append_message_state(session_id, assistant_message_key(message), state)
        else:
            store.save(session_id, state)
        logger.info(
            "capture: saved reasoning state for session %s (fields=%s)",
            session_id,
            list(state.keys()),
        )
    else:
        logger.debug("capture: no reasoning state in response for session %s", session_id)


def assistant_message_key(message: dict[str, Any]) -> str:
    """Return a stable fingerprint for an assistant message."""
    canonical = json.dumps(
        _assistant_payload(message, normalize="strict", include_tools=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def assistant_message_key_variants(message: dict[str, Any]) -> list[str]:
    """Return normalized and legacy fingerprints for compatibility."""
    normalized = assistant_message_key(message)
    legacy = hashlib.sha256(
        json.dumps(
            _assistant_payload(message, normalize="legacy", include_tools=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    loose = hashlib.sha256(
        json.dumps(
            _assistant_payload(message, normalize="loose", include_tools=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    content_only = hashlib.sha256(
        json.dumps(
            _assistant_payload(message, normalize="strict", include_tools=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    content_only_loose = hashlib.sha256(
        json.dumps(
            _assistant_payload(message, normalize="loose", include_tools=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    return [
        key
        for key in dict.fromkeys([normalized, legacy, loose, content_only, content_only_loose])
        if key
    ]


def _extract_assistant_message(response: dict[str, Any]) -> dict[str, Any] | None:
    choices = response.get("choices", [])
    if not choices:
        return None
    message = choices[0].get("message")
    return message if isinstance(message, dict) else None


def _normalize_message_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_normalize_message_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_message_value(item) for key, item in value.items()}
    return value


def _assistant_payload(message: dict[str, Any], *, normalize: str, include_tools: bool) -> dict[str, Any]:
    content = message.get("content")
    if normalize == "legacy":
        normalized_content = content
    elif normalize == "loose":
        normalized_content = _normalize_message_value_loose(content)
    else:
        normalized_content = _normalize_message_value(content)

    payload: dict[str, Any] = {
        "role": message.get("role"),
        "content": normalized_content,
        "name": message.get("name"),
    }
    if include_tools:
        payload["tool_calls"] = message.get("tool_calls")
        payload["function_call"] = message.get("function_call")
    else:
        payload["tool_calls"] = None
        payload["function_call"] = None
    return payload


def _normalize_message_value_loose(value: Any) -> Any:
    if isinstance(value, str):
        collapsed = " ".join(value.replace("\r\n", "\n").split())
        return collapsed.strip()
    if isinstance(value, list):
        return [_normalize_message_value_loose(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_message_value_loose(item) for key, item in value.items()}
    return value
