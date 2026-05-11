"""Anthropic provider adapter.

Claude models with extended thinking return ``thinking`` content blocks that
must be passed back verbatim on the next request.  This adapter translates
between the OpenAI-compatible format used by Android Studio and Anthropic's
native Messages API.

Thinking mode is controlled via the provider config:
  ``thinking_enabled``       — toggle extended thinking on / off.
  ``thinking_budget_tokens`` — token budget for the thinking block (default 5000).
"""

from __future__ import annotations

import copy
from typing import Any

from aslor.providers.base import ProviderAdapter


# Claude models that support extended thinking.
_THINKING_MODELS = (
    "claude-3-7",
    "claude-3-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5",
)


class AnthropicAdapter(ProviderAdapter):
    def __init__(
        self,
        base_url: str = "https://api.anthropic.com/v1",
        thinking_enabled: bool = False,
        thinking_budget_tokens: int = 5000,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._thinking_enabled = thinking_enabled
        self._thinking_budget = thinking_budget_tokens

    # ── config setters (so get_adapter can reconfigure without re-creating) ──

    def set_thinking(self, enabled: bool, budget_tokens: int = 5000) -> None:
        self._thinking_enabled = enabled
        self._thinking_budget = budget_tokens

    # ── ProviderAdapter interface ────────────────────────────────────────────

    def get_base_url(self) -> str:
        return self._base_url

    def get_headers(self, api_key: str) -> dict[str, str]:
        headers: dict[str, str] = {
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        if self._thinking_enabled:
            headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
        if api_key:
            headers["x-api-key"] = api_key
        return headers

    def is_reasoning_model(self, model: str) -> bool:
        """A model *requires* state repair only when thinking is enabled."""
        if not self._thinking_enabled:
            return False
        lower = model.lower()
        return any(lower.startswith(prefix) for prefix in _THINKING_MODELS)

    def reasoning_field(self) -> str:
        return "thinking" if self._thinking_enabled else ""

    def normalize_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI-style request to Anthropic native format."""
        body = copy.deepcopy(body)
        messages = body.get("messages", [])
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            thinking = msg.get("thinking")

            if role in {"system", "developer"}:
                system_parts.append(str(content))
                continue

            if role == "assistant":
                blocks: list[dict[str, Any]] = []
                if thinking:
                    if isinstance(thinking, list):
                        blocks.extend(thinking)
                    else:
                        blocks.append({"type": "thinking", "thinking": thinking})
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                anthropic_messages.append(
                    {"role": "assistant", "content": blocks or str(content)}
                )
            else:
                if isinstance(content, str):
                    anthropic_messages.append({"role": role, "content": content})
                else:
                    anthropic_messages.append({"role": role, "content": content})

        anthropic_body: dict[str, Any] = {
            "model": body["model"],
            "messages": anthropic_messages,
            "max_tokens": body.get("max_tokens", 8192),
        }
        if system_parts:
            anthropic_body["system"] = " ".join(system_parts)
        if body.get("stream"):
            anthropic_body["stream"] = True

        # Only add the thinking block when the toggle is on AND the model
        # supports it.
        if self._thinking_enabled and self._is_thinking_model(body.get("model", "")):
            anthropic_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            }
        return anthropic_body

    def extract_reasoning_state(self, response: dict[str, Any]) -> dict[str, Any] | None:
        content_blocks = response.get("content", [])
        thinking_blocks = [
            b for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        if thinking_blocks:
            return {"thinking": thinking_blocks}
        return None

    def inject_reasoning_state(
        self, messages: list[dict[str, Any]], state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        thinking = state.get("thinking")
        if not thinking:
            return messages
        result: list[dict[str, Any]] = []
        for msg in messages:
            m = copy.deepcopy(msg)
            if m.get("role") == "assistant" and not m.get("thinking"):
                m["thinking"] = thinking
            result.append(m)
        return result

    # ── internal ────────────────────────────────────────────────────────────

    def _is_thinking_model(self, model: str) -> bool:
        lower = model.lower()
        return any(lower.startswith(prefix) for prefix in _THINKING_MODELS)
