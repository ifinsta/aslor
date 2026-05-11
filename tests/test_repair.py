"""Tests for ReasoningRepairAgent."""

from __future__ import annotations

import pytest
from aslor.agents.capture import assistant_message_key
from aslor.providers.openai import OpenAIAdapter
from aslor.providers.anthropic import AnthropicAdapter
from aslor.providers.passthrough import PassthroughAdapter
from aslor.reasoning.repair import repair_messages, _has_assistant_missing_field
from aslor.reasoning.state import ReasoningStateStore


class TestHasAssistantMissingField:
    def test_detects_missing_field(self):
        messages = [{"role": "assistant", "content": "hi"}]
        assert _has_assistant_missing_field(messages, "reasoning_content") is True

    def test_no_trigger_when_field_present(self):
        messages = [{"role": "assistant", "content": "hi", "reasoning_content": "think"}]
        assert _has_assistant_missing_field(messages, "reasoning_content") is False

    def test_no_assistant_messages(self):
        messages = [{"role": "user", "content": "hello"}]
        assert _has_assistant_missing_field(messages, "reasoning_content") is False

    def test_empty_messages(self):
        assert _has_assistant_missing_field([], "reasoning_content") is False


class TestRepairMessages:
    def test_injects_when_state_cached(self, store: ReasoningStateStore):
        store.save("session1", {"reasoning_content": "I reasoned"})
        adapter = OpenAIAdapter()
        messages = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        result = repair_messages(messages, "session1", adapter, store)
        assert result[1]["reasoning_content"] == "I reasoned"

    def test_injects_latest_state_into_all_missing_assistant_messages_when_no_exact_history(self, store: ReasoningStateStore):
        store.save("session-all", {"reasoning_content": "cached"})
        adapter = OpenAIAdapter()
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        result = repair_messages(messages, "session-all", adapter, store)
        assert result[1]["reasoning_content"] == "cached"
        assert result[3]["reasoning_content"] == "cached"

    def test_passthrough_when_no_cached_state(self, store: ReasoningStateStore):
        adapter = OpenAIAdapter()
        messages = [{"role": "assistant", "content": "a"}]
        result = repair_messages(messages, "no_such_session", adapter, store)
        assert result == messages

    def test_passthrough_adapter_skips_repair(self, store: ReasoningStateStore):
        store.save("session1", {"reasoning_content": "secret"})
        adapter = PassthroughAdapter()
        messages = [{"role": "assistant", "content": "a"}]
        result = repair_messages(messages, "session1", adapter, store)
        assert "reasoning_content" not in result[0]

    def test_no_repair_when_field_already_present(self, store: ReasoningStateStore):
        store.save("session1", {"reasoning_content": "cached"})
        adapter = OpenAIAdapter()
        messages = [{"role": "assistant", "content": "a", "reasoning_content": "existing"}]
        result = repair_messages(messages, "session1", adapter, store)
        assert result[0]["reasoning_content"] == "existing"

    def test_overrides_incorrect_existing_reasoning_when_exact_history_exists(self, store: ReasoningStateStore):
        adapter = OpenAIAdapter()
        assistant = {"role": "assistant", "content": "Turn 1 answer"}
        store.append_message_state(
            "session-mismatch",
            assistant_message_key(assistant),
            {"reasoning_content": "Correct cached reasoning"},
        )
        messages = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Turn 1 answer", "reasoning_content": "Wrong replayed reasoning"},
        ]

        result = repair_messages(messages, "session-mismatch", adapter, store)

        assert result[1]["reasoning_content"] == "Correct cached reasoning"

    def test_anthropic_thinking_repair(self, store: ReasoningStateStore):
        thinking_blocks = [{"type": "thinking", "thinking": "my thoughts"}]
        store.save("s", {"thinking": thinking_blocks})
        adapter = AnthropicAdapter(thinking_enabled=True)
        messages = [{"role": "assistant", "content": "answer"}]
        result = repair_messages(messages, "s", adapter, store)
        assert result[0]["thinking"] == thinking_blocks

    def test_repairs_multiple_assistant_messages_from_history(self, store: ReasoningStateStore):
        adapter = OpenAIAdapter()
        assistant1 = {"role": "assistant", "content": "Turn 1 answer"}
        assistant2 = {"role": "assistant", "content": "Turn 2 answer"}
        store.append_message_state(
            "session-history",
            assistant_message_key(assistant1),
            {"reasoning_content": "Reasoning for turn 1"},
        )
        store.append_message_state(
            "session-history",
            assistant_message_key(assistant2),
            {"reasoning_content": "Reasoning for turn 2"},
        )
        messages = [
            {"role": "user", "content": "Question 1"},
            assistant1,
            {"role": "user", "content": "Question 2"},
            assistant2,
            {"role": "user", "content": "Question 3"},
        ]

        result = repair_messages(messages, "session-history", adapter, store)

        assert result[1]["reasoning_content"] == "Reasoning for turn 1"
        assert result[3]["reasoning_content"] == "Reasoning for turn 2"

    def test_repairs_by_assistant_turn_order_when_content_drifted(self, store: ReasoningStateStore):
        adapter = OpenAIAdapter()
        original_assistant1 = {"role": "assistant", "content": "Turn 1 answer"}
        original_assistant2 = {"role": "assistant", "content": "Turn 2 answer"}
        store.append_message_state(
            "session-order-fallback",
            assistant_message_key(original_assistant1),
            {"reasoning_content": "Reasoning for turn 1"},
        )
        store.append_message_state(
            "session-order-fallback",
            assistant_message_key(original_assistant2),
            {"reasoning_content": "Reasoning for turn 2"},
        )
        messages = [
            {"role": "user", "content": "Question 1"},
            {"role": "assistant", "content": "Turn 1 answer\n"},
            {"role": "user", "content": "Question 2"},
            {"role": "assistant", "content": "Turn 2 answer (replayed)"},
            {"role": "user", "content": "Question 3"},
        ]

        result = repair_messages(messages, "session-order-fallback", adapter, store)

        assert result[1]["reasoning_content"] == "Reasoning for turn 1"
        assert result[3]["reasoning_content"] == "Reasoning for turn 2"

    def test_recovers_exact_message_state_from_another_session(self, store: ReasoningStateStore):
        adapter = OpenAIAdapter()
        assistant = {"role": "assistant", "content": "Turn 1 answer"}
        store.append_message_state(
            "session-original",
            assistant_message_key(assistant),
            {"reasoning_content": "Reasoning from original session"},
        )
        messages = [
            {"role": "user", "content": "Same opener"},
            {"role": "assistant", "content": "Turn 1 answer"},
            {"role": "user", "content": "Follow up"},
        ]

        result = repair_messages(messages, "session-collided", adapter, store)

        assert result[1]["reasoning_content"] == "Reasoning from original session"
