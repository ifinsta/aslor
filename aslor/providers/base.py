"""Abstract base class for all provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ProviderAdapter(ABC):
    """Contract every provider adapter must fulfill."""

    @abstractmethod
    def normalize_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Transform the request body into the format expected by this provider."""

    @abstractmethod
    def is_reasoning_model(self, model: str) -> bool:
        """Return True if the model requires reasoning-state injection."""

    @abstractmethod
    def extract_reasoning_state(self, response: dict[str, Any]) -> dict[str, Any] | None:
        """Extract reasoning state fields from a completed (non-streaming) response.

        Returns a dict like ``{"reasoning_content": "..."}`` or ``None`` if no
        reasoning state was present.
        """

    @abstractmethod
    def inject_reasoning_state(
        self, messages: list[dict[str, Any]], state: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Re-inject cached reasoning state into the assistant turn that is
        missing it.

        Returns a new list with the updated messages; never mutates the input.
        """

    @abstractmethod
    def get_headers(self, api_key: str) -> dict[str, str]:
        """Return HTTP headers required by this provider."""

    @abstractmethod
    def get_base_url(self) -> str:
        """Return the provider's API base URL (no trailing slash)."""

    def reasoning_field(self) -> str:
        """Name of the field that holds reasoning state on assistant messages."""
        return "reasoning_content"

    @staticmethod
    def _strip_multimodal_content(messages: list[dict[str, Any]]) -> None:
        """Strip ``image_url`` content parts from messages in-place.

        Text-only providers (DeepSeek, non-vision OpenAI models) reject
        ``image_url`` content parts.  This helper converts content arrays
        to plain strings, replacing image parts with an ``[image]``
        placeholder so the text context is preserved.
        """
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            parts: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif part.get("type") == "image_url":
                    parts.append("[image]")
            msg["content"] = "\n".join(parts) if parts else ""

    @staticmethod
    def _map_roles(messages: list[dict[str, Any]], mapping: dict[str, str]) -> None:
        for msg in messages:
            role = msg.get("role")
            if isinstance(role, str) and role in mapping:
                msg["role"] = mapping[role]
