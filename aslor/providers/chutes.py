"""Chutes provider adapter.

Chutes exposes OpenAI-compatible ``/v1/chat/completions`` endpoints for many
different models. Unlike the generic passthrough adapter, this adapter keeps
multimodal content arrays intact so vision-language Chutes models can receive
image inputs directly.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from aslor.models.registry import get_capability, get_registry
from aslor.providers.base import ProviderAdapter


class ChutesAdapter(ProviderAdapter):
    def __init__(self, base_url: str = "https://llm.chutes.ai/v1") -> None:
        self._base_url = base_url.rstrip("/")

    def get_base_url(self) -> str:
        return self._base_url

    def get_headers(self, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def is_reasoning_model(self, model: str) -> bool:
        return get_capability(model).reasoning

    def reasoning_field(self) -> str:
        return "reasoning_content"

    def normalize_request(self, body: dict[str, Any]) -> dict[str, Any]:
        body = copy.deepcopy(body)
        messages = body.get("messages", [])
        self._map_roles(messages, {"developer": "system"})
        model = body.get("model", "")
        known_model = get_registry().find_model(str(model)) is not None
        is_reasoning = self.is_reasoning_model(str(model)) or _body_indicates_reasoning(body)
        if not is_reasoning and not known_model and _messages_contain_reasoning(messages):
            is_reasoning = True
        if not is_reasoning:
            for msg in messages:
                msg.pop("reasoning_content", None)
                msg.pop("thinking", None)
        return body

    def extract_reasoning_state(self, response: dict[str, Any]) -> dict[str, Any] | None:
        choices = response.get("choices", [])
        if not choices:
            return None
        message = choices[0].get("message", {})
        rc = message.get("reasoning_content")
        if rc:
            return {"reasoning_content": rc}
        return None

    def inject_reasoning_state(
        self, messages: list[dict[str, Any]], state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        rc = state.get("reasoning_content")
        if not rc:
            return messages
        result = []
        for msg in messages:
            m = copy.deepcopy(msg)
            if m.get("role") == "assistant" and not m.get("reasoning_content"):
                m["reasoning_content"] = rc
            result.append(m)
        return result


def _body_indicates_reasoning(body: dict[str, Any]) -> bool:
    if "reasoning_effort" in body or "reasoning" in body:
        return True
    model = str(body.get("model", "")).lower()
    if re.search(r"(^|[^a-z0-9])o(1|3|4)([^a-z0-9]|$)", model):
        return True
    if "deepseek" in model and ("reasoner" in model or "r1" in model):
        return True
    return False


def _messages_contain_reasoning(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for m in messages:
        if isinstance(m, dict) and ("reasoning_content" in m or "thinking" in m):
            return True
    return False
