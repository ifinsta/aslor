"""Tests for ResponseCaptureAgent."""

from __future__ import annotations

import pytest
from aslor.agents.capture import assistant_message_key, capture_from_assembled
from aslor.providers.openai import OpenAIAdapter
from aslor.providers.deepseek import DeepSeekAdapter
from aslor.providers.passthrough import PassthroughAdapter
from aslor.reasoning.state import ReasoningStateStore


class TestCaptureFromAssembled:
    def test_saves_reasoning_content(self, store: ReasoningStateStore):
        adapter = OpenAIAdapter()
        response = {
            "choices": [{"message": {"role": "assistant", "content": "ans", "reasoning_content": "thought"}}]
        }
        capture_from_assembled(response, "sess1", adapter, store)
        state = store.load("sess1")
        assert state == {"reasoning_content": "thought"}

    def test_no_save_when_no_reasoning(self, store: ReasoningStateStore):
        adapter = OpenAIAdapter()
        response = {"choices": [{"message": {"role": "assistant", "content": "ans"}}]}
        capture_from_assembled(response, "sess2", adapter, store)
        assert store.load("sess2") is None

    def test_passthrough_never_saves(self, store: ReasoningStateStore):
        adapter = PassthroughAdapter()
        response = {"choices": [{"message": {"role": "assistant", "reasoning_content": "secret"}}]}
        capture_from_assembled(response, "sess3", adapter, store)
        assert store.load("sess3") is None

    def test_deepseek_saves_reasoning(self, store: ReasoningStateStore):
        adapter = DeepSeekAdapter()
        response = {"choices": [{"message": {"reasoning_content": "deep think", "content": "hi"}}]}
        capture_from_assembled(response, "dsess", adapter, store)
        assert store.load("dsess") == {"reasoning_content": "deep think"}

    def test_overwrites_previous_state(self, store: ReasoningStateStore):
        adapter = OpenAIAdapter()
        store.save("s", {"reasoning_content": "old"})
        response = {"choices": [{"message": {"reasoning_content": "new", "content": "x"}}]}
        capture_from_assembled(response, "s", adapter, store)
        assert store.load("s") == {"reasoning_content": "new"}

    def test_preserves_per_message_history(self, store: ReasoningStateStore):
        adapter = OpenAIAdapter()
        response1 = {
            "choices": [{"message": {"role": "assistant", "content": "a1", "reasoning_content": "r1"}}]
        }
        response2 = {
            "choices": [{"message": {"role": "assistant", "content": "a2", "reasoning_content": "r2"}}]
        }

        capture_from_assembled(response1, "history", adapter, store)
        capture_from_assembled(response2, "history", adapter, store)

        history = store.load_history("history")
        states_by_key = {item["assistant_key"]: item["state"] for item in history}
        assert states_by_key[assistant_message_key(response1["choices"][0]["message"])] == {"reasoning_content": "r1"}
        assert states_by_key[assistant_message_key(response2["choices"][0]["message"])] == {"reasoning_content": "r2"}
        assert store.load("history") == {"reasoning_content": "r2"}
