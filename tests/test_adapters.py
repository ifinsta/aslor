"""Tests for all provider adapters."""

from __future__ import annotations

import pytest
from aslor.providers.openai import OpenAIAdapter
from aslor.providers.deepseek import DeepSeekAdapter
from aslor.providers.anthropic import AnthropicAdapter
from aslor.providers.chutes import ChutesAdapter
from aslor.providers.passthrough import PassthroughAdapter


class TestOpenAIAdapter:
    def setup_method(self):
        self.adapter = OpenAIAdapter()

    def test_is_reasoning_model_o1(self):
        assert self.adapter.is_reasoning_model("o1-mini") is True

    def test_is_reasoning_model_o3(self):
        assert self.adapter.is_reasoning_model("o3") is True

    def test_is_reasoning_model_o4(self):
        assert self.adapter.is_reasoning_model("o4-mini") is True

    def test_not_reasoning_gpt4(self):
        assert self.adapter.is_reasoning_model("gpt-4o") is False

    def test_reasoning_field(self):
        assert self.adapter.reasoning_field() == "reasoning_content"

    def test_extract_state_present(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "hi", "reasoning_content": "think"}}]}
        state = self.adapter.extract_reasoning_state(resp)
        assert state == {"reasoning_content": "think"}

    def test_extract_state_absent(self):
        resp = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
        assert self.adapter.extract_reasoning_state(resp) is None

    def test_extract_state_empty_choices(self):
        assert self.adapter.extract_reasoning_state({"choices": []}) is None

    def test_inject_missing_reasoning(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        state = {"reasoning_content": "I thought about it"}
        result = self.adapter.inject_reasoning_state(msgs, state)
        assert result[1]["reasoning_content"] == "I thought about it"

    def test_inject_does_not_overwrite_existing(self):
        msgs = [{"role": "assistant", "content": "x", "reasoning_content": "original"}]
        result = self.adapter.inject_reasoning_state(msgs, {"reasoning_content": "new"})
        assert result[0]["reasoning_content"] == "original"

    def test_inject_does_not_mutate_input(self):
        msgs = [{"role": "assistant", "content": "x"}]
        original = msgs[0].copy()
        self.adapter.inject_reasoning_state(msgs, {"reasoning_content": "t"})
        assert msgs[0] == original

    def test_normalize_strips_reasoning_for_non_reasoning_model(self):
        body = {
            "model": "gpt-4o",
            "messages": [{"role": "assistant", "content": "hi", "reasoning_content": "leak"}],
        }
        result = self.adapter.normalize_request(body)
        assert "reasoning_content" not in result["messages"][0]

    def test_normalize_keeps_reasoning_for_reasoning_model(self):
        body = {
            "model": "o1-mini",
            "messages": [{"role": "assistant", "content": "hi", "reasoning_content": "think"}],
        }
        result = self.adapter.normalize_request(body)
        assert result["messages"][0]["reasoning_content"] == "think"

    def test_normalize_keeps_reasoning_when_body_indicates_reasoning_mode(self):
        body = {
            "model": "my-custom-deployment",
            "reasoning_effort": "high",
            "messages": [{"role": "assistant", "content": "hi", "reasoning_content": "think"}],
        }
        result = self.adapter.normalize_request(body)
        assert result["messages"][0]["reasoning_content"] == "think"

    def test_get_headers(self):
        headers = self.adapter.get_headers("sk-test")
        assert headers["Authorization"] == "Bearer sk-test"
        assert "Content-Type" in headers

    def test_get_base_url(self):
        assert self.adapter.get_base_url() == "https://api.openai.com/v1"


class TestDeepSeekAdapter:
    def setup_method(self):
        self.adapter = DeepSeekAdapter()

    def test_is_reasoning_model(self):
        assert self.adapter.is_reasoning_model("deepseek-reasoner") is True
        assert self.adapter.is_reasoning_model("deepseek-r1-lite") is True

    def test_not_reasoning(self):
        assert self.adapter.is_reasoning_model("deepseek-chat") is False

    def test_extract_state(self):
        resp = {"choices": [{"message": {"reasoning_content": "chain"}}]}
        assert self.adapter.extract_reasoning_state(resp) == {"reasoning_content": "chain"}

    def test_inject_state(self):
        msgs = [{"role": "assistant", "content": "a"}]
        result = self.adapter.inject_reasoning_state(msgs, {"reasoning_content": "rc"})
        assert result[0]["reasoning_content"] == "rc"


class TestAnthropicAdapter:
    def setup_method(self):
        self.adapter = AnthropicAdapter(thinking_enabled=True, thinking_budget_tokens=5000)

    def test_is_reasoning_model_when_thinking_on(self):
        assert self.adapter.is_reasoning_model("claude-3-7-sonnet-20250219") is True

    def test_is_reasoning_model_when_thinking_off(self):
        off = AnthropicAdapter(thinking_enabled=False)
        assert off.is_reasoning_model("claude-3-7-sonnet-20250219") is False

    def test_reasoning_model_claude_35(self):
        assert self.adapter.is_reasoning_model("claude-3-5-sonnet") is True

    def test_reasoning_field_when_thinking_on(self):
        assert self.adapter.reasoning_field() == "thinking"

    def test_reasoning_field_when_thinking_off(self):
        off = AnthropicAdapter(thinking_enabled=False)
        assert off.reasoning_field() == ""

    def test_extract_thinking_state(self):
        resp = {
            "content": [
                {"type": "thinking", "thinking": "my thoughts"},
                {"type": "text", "text": "answer"},
            ]
        }
        state = self.adapter.extract_reasoning_state(resp)
        assert state is not None
        assert len(state["thinking"]) == 1
        assert state["thinking"][0]["type"] == "thinking"

    def test_extract_no_thinking(self):
        resp = {"content": [{"type": "text", "text": "answer"}]}
        assert self.adapter.extract_reasoning_state(resp) is None

    def test_inject_thinking(self):
        msgs = [{"role": "assistant", "content": "hi"}]
        state = {"thinking": [{"type": "thinking", "thinking": "x"}]}
        result = self.adapter.inject_reasoning_state(msgs, state)
        assert result[0]["thinking"] == state["thinking"]

    def test_normalize_converts_to_anthropic_format(self):
        body = {
            "model": "claude-3-7-sonnet-20250219",
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
            "max_tokens": 1024,
        }
        result = self.adapter.normalize_request(body)
        assert result["system"] == "Be helpful"
        assert result["messages"][0]["role"] == "user"
        assert "thinking" in result

    def test_get_headers_contains_version(self):
        headers = self.adapter.get_headers("key")
        assert "anthropic-version" in headers
        assert headers["x-api-key"] == "key"


class TestPassthroughAdapter:
    def setup_method(self):
        self.adapter = PassthroughAdapter()

    def test_never_reasoning(self):
        assert self.adapter.is_reasoning_model("any-model") is False

    def test_extract_returns_none(self):
        assert self.adapter.extract_reasoning_state({"choices": [{"message": {"reasoning_content": "x"}}]}) is None

    def test_inject_returns_unchanged(self):
        msgs = [{"role": "assistant", "content": "x"}]
        result = self.adapter.inject_reasoning_state(msgs, {"reasoning_content": "rc"})
        assert result == msgs

    def test_normalize_strips_all_reasoning_fields(self):
        body = {
            "model": "llama3",
            "messages": [{"role": "assistant", "content": "hi", "reasoning_content": "rc", "thinking": "t"}],
        }
        result = self.adapter.normalize_request(body)
        msg = result["messages"][0]
        assert "reasoning_content" not in msg
        assert "thinking" not in msg

    def test_normalize_maps_developer_role_to_system(self):
        body = {
            "model": "llama3",
            "messages": [{"role": "developer", "content": "do this"}, {"role": "user", "content": "hi"}],
        }
        result = self.adapter.normalize_request(body)
        assert result["messages"][0]["role"] == "system"


class TestDeepSeekAdapterRoleNormalization:
    def test_normalize_maps_developer_role_to_system(self):
        adapter = DeepSeekAdapter()
        body = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "developer", "content": "do this"}, {"role": "user", "content": "hi"}],
        }
        result = adapter.normalize_request(body)
        assert result["messages"][0]["role"] == "system"


class TestChutesAdapterRoleNormalization:
    def test_normalize_maps_developer_role_to_system(self):
        adapter = ChutesAdapter()
        body = {
            "model": "Qwen/Qwen3-32B-TEE",
            "messages": [{"role": "developer", "content": "do this"}, {"role": "user", "content": "hi"}],
        }
        result = adapter.normalize_request(body)
        assert result["messages"][0]["role"] == "system"


class TestChutesAdapter:
    def setup_method(self):
        self.adapter = ChutesAdapter()

    def test_never_strips_multimodal_content(self):
        body = {
            "model": "Qwen/Qwen2.5-VL-32B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                    ],
                }
            ],
        }
        result = self.adapter.normalize_request(body)
        assert isinstance(result["messages"][0]["content"], list)
        assert result["messages"][0]["content"][1]["type"] == "image_url"

    def test_strips_reasoning_for_non_reasoning_model(self):
        body = {
            "model": "Qwen/Qwen3-32B-TEE",
            "messages": [{"role": "assistant", "content": "hi", "reasoning_content": "secret"}],
        }
        result = self.adapter.normalize_request(body)
        assert "reasoning_content" not in result["messages"][0]

    def test_get_headers(self):
        headers = self.adapter.get_headers("sk-chutes")
        assert headers["Authorization"] == "Bearer sk-chutes"
        assert headers["Content-Type"] == "application/json"
