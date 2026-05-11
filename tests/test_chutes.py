"""Tests for Chutes provider integration."""

from __future__ import annotations

import json
import os

import httpx
import pytest
import pytest_asyncio
from httpx import AsyncClient

from aslor.config import AppConfig
from aslor.providers.chutes import ChutesAdapter
from aslor.providers.registry import get_adapter
from aslor.reasoning.detector import detect
from aslor.server.app import create_app


class TestChutesAdapterSelection:
    def test_provider_name_chutes_selects_chutes_adapter(self):
        config = AppConfig(
            {
                "provider": {
                    "name": "chutes",
                    "base_url": "https://llm.chutes.test/v1",
                    "api_key_env": "CHUTES_API_KEY",
                },
            }
        )
        adapter = get_adapter(config)
        assert isinstance(adapter, ChutesAdapter)
        assert adapter.get_base_url() == "https://llm.chutes.test/v1"

    def test_model_registry_hint_selects_chutes_adapter(self):
        config = AppConfig(
            {
                "provider": {
                    "name": "",
                    "base_url": "https://llm.chutes.test/v1",
                    "api_key_env": "CHUTES_API_KEY",
                },
            }
        )
        adapter = get_adapter(config, model_name="Qwen/Qwen3-32B-TEE")
        assert isinstance(adapter, ChutesAdapter)


class TestChutesDetection:
    @pytest.fixture
    def cfg(self):
        return AppConfig({"provider": {"name": "chutes", "base_url": "https://llm.chutes.ai/v1"}})

    def test_known_chutes_model_detected(self, cfg):
        body = {"model": "Qwen/Qwen3-32B-TEE", "messages": [{"role": "user", "content": "hi"}]}
        result = detect(body, cfg)
        assert result.provider == "chutes"
        assert result.needs_repair is False


@pytest.fixture
def chutes_app_config(tmp_path):
    os.environ["CHUTES_TEST_KEY"] = "sk-chutes-test"
    return AppConfig(
        {
            "server": {"host": "127.0.0.1", "port": 8080},
            "provider": {
                "name": "chutes",
                "base_url": "https://llm.chutes.test/v1",
                "api_key_env": "CHUTES_TEST_KEY",
                "timeout_seconds": 10,
            },
            "cache": {"path": str(tmp_path / "chutes_test.db"), "encrypt": False},
            "logging": {"level": "WARNING", "format": "text"},
        }
    )


@pytest_asyncio.fixture
async def chutes_client(chutes_app_config: AppConfig):
    app = create_app(chutes_app_config)
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_chutes_preserves_multimodal_payload(chutes_client: AsyncClient, httpx_mock):
    captured_bodies: list[dict] = []
    captured_headers: list[dict] = []

    def capture_request(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        captured_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-chutes",
                "object": "chat.completion",
                "model": "Qwen/Qwen2.5-VL-32B-Instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "I can see the screenshot."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(capture_request, method="POST", url="https://llm.chutes.test/v1/chat/completions")

    payload = {
        "model": "Qwen/Qwen2.5-VL-32B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this screen"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/screen.png"}},
                ],
            }
        ],
        "stream": False,
    }
    resp = await chutes_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert len(captured_bodies) == 1
    assert isinstance(captured_bodies[0]["messages"][0]["content"], list)
    assert captured_bodies[0]["messages"][0]["content"][1]["type"] == "image_url"
    assert captured_headers[0]["authorization"] == "Bearer sk-chutes-test"
