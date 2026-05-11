"""Pydantic schemas for incoming OpenAI-compatible chat completion requests."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    reasoning_content: str | None = None
    thinking: Any | None = None

    model_config = {"extra": "allow"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | str | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    model_config = {"extra": "allow"}

    def to_dict(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True)
        return data
