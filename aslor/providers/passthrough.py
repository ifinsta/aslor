"""Passthrough adapter — safe forward for any OpenAI-compatible provider.

No reasoning-state manipulation; the request is forwarded as-is.
Used for local models (Ollama, LM Studio), OpenRouter, or any provider whose
models do not require reasoning state.
"""

from __future__ import annotations

import copy
from typing import Any

from aslor.providers.base import ProviderAdapter


class PassthroughAdapter(ProviderAdapter):
    def __init__(self, base_url: str = "http://localhost:11434/v1") -> None:
        self._base_url = base_url.rstrip("/")

    def get_base_url(self) -> str:
        return self._base_url

    def get_headers(self, api_key: str) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def is_reasoning_model(self, model: str) -> bool:
        return False

    def reasoning_field(self) -> str:
        return ""

    def normalize_request(self, body: dict[str, Any]) -> dict[str, Any]:
        body = copy.deepcopy(body)
        messages = body.get("messages", [])
        self._map_roles(messages, {"developer": "system"})
        for msg in messages:
            msg.pop("reasoning_content", None)
            msg.pop("thinking", None)
        self._strip_multimodal_content(messages)
        return body

    def extract_reasoning_state(self, response: dict[str, Any]) -> dict[str, Any] | None:
        return None

    def inject_reasoning_state(
        self, messages: list[dict[str, Any]], state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return messages
