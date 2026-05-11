"""Tests for ModelDetectorAgent."""

from __future__ import annotations

import pytest
from aslor.reasoning.detector import detect, _derive_session_key
from aslor.config import AppConfig


@pytest.fixture
def cfg():
    return AppConfig({"provider": {"name": "openai", "base_url": "https://api.openai.com/v1"}})


class TestDetect:
    def test_detects_reasoning_model_o1(self, cfg: AppConfig):
        body = {"model": "o1-mini", "messages": [{"role": "user", "content": "hi"}]}
        result = detect(body, cfg)
        assert result.needs_repair is True
        assert result.reasoning_field == "reasoning_content"

    def test_detects_deepseek_reasoner(self, cfg: AppConfig):
        body = {"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "hi"}]}
        result = detect(body, cfg)
        assert result.needs_repair is True

    def test_detects_android_studio_deepseek_v4_flash(self, cfg: AppConfig):
        body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "hi"}]}
        result = detect(body, cfg)
        assert result.needs_repair is True
        assert result.reasoning_field == "reasoning_content"

    def test_no_repair_for_gpt4(self, cfg: AppConfig):
        body = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        result = detect(body, cfg)
        assert result.needs_repair is False

    def test_session_key_is_deterministic(self, cfg: AppConfig):
        body = {"model": "o1-mini", "messages": [{"role": "user", "content": "same question"}]}
        k1 = detect(body, cfg).session_key
        k2 = detect(body, cfg).session_key
        assert k1 == k2

    def test_detects_reasoning_mode_from_reasoning_effort_even_when_model_unknown(self, cfg: AppConfig):
        body = {
            "model": "my-custom-deployment",
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = detect(body, cfg)
        assert result.needs_repair is True
        assert result.reasoning_field == "reasoning_content"

    def test_detects_reasoning_mode_from_model_token_not_prefix(self, cfg: AppConfig):
        body = {
            "model": "azure-deploy-o1-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = detect(body, cfg)
        assert result.needs_repair is True
        assert result.reasoning_field == "reasoning_content"

    def test_session_key_prefers_stable_conversation_metadata(self, cfg: AppConfig):
        body1 = {
            "model": "o1",
            "metadata": {"conversation_id": "conv-123"},
            "messages": [{"role": "user", "content": "Question A"}],
        }
        body2 = {
            "model": "o1",
            "metadata": {"conversation_id": "conv-123"},
            "messages": [{"role": "user", "content": "Question B"}],
        }
        assert detect(body1, cfg).session_key == detect(body2, cfg).session_key

    def test_different_questions_different_keys(self, cfg: AppConfig):
        body1 = {"model": "o1", "messages": [{"role": "user", "content": "question A"}]}
        body2 = {"model": "o1", "messages": [{"role": "user", "content": "question B"}]}
        k1 = detect(body1, cfg).session_key
        k2 = detect(body2, cfg).session_key
        assert k1 != k2

    def test_session_key_length(self, cfg: AppConfig):
        body = {"model": "o1", "messages": [{"role": "user", "content": "q"}]}
        key = detect(body, cfg).session_key
        assert len(key) == 32


class TestDeriveSessionKey:
    def test_empty_messages(self):
        key = _derive_session_key({"model": "o1", "messages": []})
        assert len(key) == 32

    def test_no_user_message(self):
        key = _derive_session_key({"model": "o1", "messages": [{"role": "system", "content": "sys"}]})
        assert len(key) == 32
