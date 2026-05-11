"""Pydantic schemas for outgoing OpenAI-compatible chat completion responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None
    reasoning_content: str | None = None

    model_config = {"extra": "allow"}


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: str | None = None

    model_config = {"extra": "allow"}


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[Choice]
    usage: Usage | None = None

    model_config = {"extra": "allow"}
