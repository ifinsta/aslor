"""Integration and unit tests for DeepSeek provider — reasoning state
round-trips, streaming, non-streaming, adapter registry selection, and edge cases."""

from __future__ import annotations

import json
import os
import pytest
import pytest_asyncio
import httpx
from httpx import AsyncClient

from aslor.config import AppConfig
from aslor.providers.deepseek import DeepSeekAdapter
from aslor.providers.registry import get_adapter
from aslor.reasoning.detector import detect
from aslor.reasoning.repair import repair_messages
from aslor.reasoning.state import ReasoningStateStore
from aslor.server.app import create_app


# ── Adapter unit tests ──────────────────────────────────────────────────────

class TestDeepSeekAdapterDeep:
    """Additional adapter tests beyond the basic ones in test_adapters.py."""

    def setup_method(self):
        self.adapter = DeepSeekAdapter()

    def test_reasoner_prefix_matches_r1(self):
        assert self.adapter.is_reasoning_model("deepseek-r1") is True
        assert self.adapter.is_reasoning_model("deepseek-r1-lite") is True
        assert self.adapter.is_reasoning_model("deepseek-r1-distill-llama-70b") is True

    def test_chat_model_not_reasoning(self):
        assert self.adapter.is_reasoning_model("deepseek-chat") is False
        assert self.adapter.is_reasoning_model("deepseek-v3") is False

    def test_extract_state_empty_message(self):
        resp = {"choices": [{"message": {}}]}
        assert self.adapter.extract_reasoning_state(resp) is None

    def test_extract_state_reasoning_none(self):
        resp = {"choices": [{"message": {"content": "hi", "reasoning_content": None}}]}
        assert self.adapter.extract_reasoning_state(resp) is None

    def test_extract_state_reasoning_empty_string(self):
        resp = {"choices": [{"message": {"reasoning_content": ""}}]}
        assert self.adapter.extract_reasoning_state(resp) is None

    def test_normalize_strips_reasoning_for_chat_model(self):
        body = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "assistant", "content": "hi", "reasoning_content": "should be stripped"}
            ],
        }
        result = self.adapter.normalize_request(body)
        assert "reasoning_content" not in result["messages"][0]

    def test_normalize_keeps_reasoning_for_reasoner(self):
        body = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "assistant", "content": "hi", "reasoning_content": "keep this"}
            ],
        }
        result = self.adapter.normalize_request(body)
        assert result["messages"][0]["reasoning_content"] == "keep this"

    def test_normalize_does_not_mutate_input(self):
        body = {
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "q"}],
        }
        orig_messages = body["messages"][0].copy()
        self.adapter.normalize_request(body)
        assert body["messages"][0] == orig_messages

    def test_base_url_default(self):
        adapter = DeepSeekAdapter()
        assert adapter.get_base_url() == "https://api.deepseek.com/v1"

    def test_base_url_custom(self):
        adapter = DeepSeekAdapter(base_url="https://custom.deepseek.com/beta")
        assert adapter.get_base_url() == "https://custom.deepseek.com/beta"

    def test_inject_reasoning_state_multiple_assistant_messages(self):
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        state = {"reasoning_content": "shared reasoning"}
        result = self.adapter.inject_reasoning_state(msgs, state)
        # All assistant messages missing reasoning_content get it injected
        assert result[1]["reasoning_content"] == "shared reasoning"
        assert result[3]["reasoning_content"] == "shared reasoning"

    def test_inject_no_duplicate_when_first_has_reasoning(self):
        msgs = [
            {"role": "assistant", "content": "a1", "reasoning_content": "existing"},
            {"role": "assistant", "content": "a2"},
        ]
        state = {"reasoning_content": "cached"}
        result = self.adapter.inject_reasoning_state(msgs, state)
        assert result[0]["reasoning_content"] == "existing"
        assert result[1]["reasoning_content"] == "cached"


# ── Registry / Adapter selection tests ───────────────────────────────────────

class TestDeepSeekAdapterSelection:
    def test_provider_name_deepseek_selects_deepseek_adapter(self):
        config = AppConfig({
            "provider": {
                "name": "deepseek",
                "base_url": "https://api.deepseek.com/v1",
                "api_key_env": "DEEPSEEK_API_KEY",
            },
        })
        adapter = get_adapter(config)
        assert isinstance(adapter, DeepSeekAdapter)
        assert adapter.get_base_url() == "https://api.deepseek.com/v1"

    def test_model_registry_hint_selects_deepseek_adapter(self):
        """When provider name is unrecognized, the model-name registry hint kicks in."""
        config = AppConfig({
            "provider": {
                "name": "",  # empty/unknown — falls through to model hint
                "base_url": "https://some-proxy.com/v1",
                "api_key_env": "KEY",
            },
        })
        adapter = get_adapter(config, model_name="deepseek-reasoner")
        assert isinstance(adapter, DeepSeekAdapter)
        assert adapter.get_base_url() == "https://some-proxy.com/v1"

    def test_passthrough_provider_uses_deepseek_adapter_for_deepseek_model(self):
        config = AppConfig({
            "provider": {
                "name": "passthrough",
                "base_url": "https://some-proxy.com/v1",
                "api_key_env": "KEY",
            },
        })
        adapter = get_adapter(config, model_name="deepseek-v4-pro")
        assert isinstance(adapter, DeepSeekAdapter)
        assert adapter.get_base_url() == "https://some-proxy.com/v1"


# ── Detector tests for DeepSeek ──────────────────────────────────────────────

class TestDeepSeekDetection:
    @pytest.fixture
    def cfg(self):
        return AppConfig({"provider": {"name": "deepseek", "base_url": "https://api.deepseek.com/v1"}})

    def test_detects_reasoner_needs_repair(self, cfg):
        body = {"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "test"}]}
        result = detect(body, cfg)
        assert result.needs_repair is True
        assert result.reasoning_field == "reasoning_content"
        assert result.provider == "deepseek"
        assert result.model_name == "deepseek-reasoner"

    def test_detects_r1_needs_repair(self, cfg):
        body = {"model": "deepseek-r1-lite", "messages": [{"role": "user", "content": "test"}]}
        result = detect(body, cfg)
        assert result.needs_repair is True

    def test_detects_chat_no_repair(self, cfg):
        body = {"model": "deepseek-chat", "messages": [{"role": "user", "content": "test"}]}
        result = detect(body, cfg)
        assert result.needs_repair is False

    def test_session_key_consistent_for_same_conversation(self, cfg):
        body = {
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "user", "content": "Solve this problem"},
                {"role": "assistant", "content": "Here is my answer"},
                {"role": "user", "content": "Now elaborate"},
            ],
        }
        key1 = detect(body, cfg).session_key
        key2 = detect(body, cfg).session_key
        assert key1 == key2

    def test_session_key_changes_with_different_first_message(self, cfg):
        body1 = {"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "Topic A"}]}
        body2 = {"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "Topic B"}]}
        assert detect(body1, cfg).session_key != detect(body2, cfg).session_key


# ── Repair round-trip with DeepSeek adapter ──────────────────────────────────

class TestDeepSeekRepair:
    def test_deepseek_adapter_repair_injects_cached_reasoning(self, store: ReasoningStateStore):
        store.save("deepseek-session", {"reasoning_content": "DeepSeek chain of thought"})
        adapter = DeepSeekAdapter()
        messages = [
            {"role": "user", "content": "Explain quantum computing"},
            {"role": "assistant", "content": "Quantum computing uses qubits..."},
            {"role": "user", "content": "Go deeper"},
        ]
        result = repair_messages(messages, "deepseek-session", adapter, store)
        assert result[1]["reasoning_content"] == "DeepSeek chain of thought"

    def test_repair_skipped_when_reasoning_already_present(self, store: ReasoningStateStore):
        store.save("ds-session", {"reasoning_content": "cached stale"})
        adapter = DeepSeekAdapter()
        messages = [
            {"role": "assistant", "content": "a", "reasoning_content": "fresh from context"},
        ]
        result = repair_messages(messages, "ds-session", adapter, store)
        assert result[0]["reasoning_content"] == "fresh from context"


# ── Full integration tests via FastAPI app ───────────────────────────────────

@pytest.fixture
def deepseek_app_config(tmp_path):
    os.environ["DEEPSEEK_TEST_KEY"] = "sk-deepseek-test"
    return AppConfig(
        {
            "server": {"host": "127.0.0.1", "port": 8080},
            "provider": {
                "name": "deepseek",
                "base_url": "https://api.deepseek.test/v1",
                "api_key_env": "DEEPSEEK_TEST_KEY",
                "timeout_seconds": 10,
            },
            "cache": {"path": str(tmp_path / "deepseek_test.db"), "encrypt": False},
            "logging": {"level": "WARNING", "format": "text"},
        }
    )


@pytest.fixture
def deepseek_passthrough_app_config(tmp_path):
    os.environ["DEEPSEEK_TEST_KEY"] = "sk-deepseek-test"
    return AppConfig(
        {
            "server": {"host": "127.0.0.1", "port": 8080},
            "provider": {
                "name": "passthrough",
                "base_url": "https://api.deepseek.test/v1",
                "api_key_env": "DEEPSEEK_TEST_KEY",
                "default_model": "deepseek-v4-pro",
                "timeout_seconds": 10,
            },
            "cache": {"path": str(tmp_path / "deepseek_passthrough_test.db"), "encrypt": False},
            "logging": {"level": "WARNING", "format": "text"},
        }
    )


@pytest_asyncio.fixture
async def deepseek_client(deepseek_app_config: AppConfig):
    app = create_app(deepseek_app_config)
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def deepseek_passthrough_client(deepseek_passthrough_app_config: AppConfig):
    app = create_app(deepseek_passthrough_app_config)
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_deepseek_status_endpoint(deepseek_client: AsyncClient):
    resp = await deepseek_client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert data["provider"] == "deepseek"
    assert data["provider_base_url"] == "https://api.deepseek.test/v1"


@pytest.mark.asyncio
async def test_deepseek_non_streaming_request(deepseek_client: AsyncClient, httpx_mock):
    upstream_response = {
        "id": "ds-chatcmpl-1",
        "object": "chat.completion",
        "model": "deepseek-reasoner",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 4.",
                    "reasoning_content": "Let me count: 2+2=4",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.test/v1/chat/completions",
        status_code=200,
        json=upstream_response,
    )
    payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "stream": False,
    }
    resp = await deepseek_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "The answer is 4."
    assert data["choices"][0]["message"]["reasoning_content"] == "Let me count: 2+2=4"


@pytest.mark.asyncio
async def test_deepseek_upstream_error_forwarded(deepseek_client: AsyncClient, httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.test/v1/chat/completions",
        status_code=400,
        json={"error": {"message": "The reasoning_content must be passed back.", "type": "invalid_request_error"}},
    )
    payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    resp = await deepseek_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Upstream error" in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_deepseek_reasoning_round_trip(deepseek_client: AsyncClient, httpx_mock, deepseek_app_config: AppConfig):
    """Turn 1: response caches reasoning state. Turn 2: missing reasoning is injected."""
    # Turn 1 response
    turn1_response = {
        "id": "ds-1",
        "object": "chat.completion",
        "model": "deepseek-reasoner",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'll analyze this code.",
                    "reasoning_content": "The code has a bug on line 42 because...",
                },
                "finish_reason": "stop",
            }
        ],
    }
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.test/v1/chat/completions",
        status_code=200,
        json=turn1_response,
    )

    turn1 = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Review this code"}],
        "stream": False,
    }
    resp1 = await deepseek_client.post("/v1/chat/completions", json=turn1)
    assert resp1.status_code == 200

    # Capture the forwarded body for turn 2
    captured_bodies = []

    def capture_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        return httpx.Response(
            200,
            json={
                "id": "ds-2",
                "object": "chat.completion",
                "model": "deepseek-reasoner",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "The fix is on line 42."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(capture_request)

    turn2 = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "user", "content": "Review this code"},
            {"role": "assistant", "content": "I'll analyze this code."},
            {"role": "user", "content": "What's the fix?"},
        ],
        "stream": False,
    }
    resp2 = await deepseek_client.post("/v1/chat/completions", json=turn2)
    assert resp2.status_code == 200

    assert len(captured_bodies) == 1
    forwarded_msgs = captured_bodies[0]["messages"]
    assistant_msg = next(m for m in forwarded_msgs if m["role"] == "assistant")
    assert assistant_msg.get("reasoning_content") == "The code has a bug on line 42 because..."


@pytest.mark.asyncio
async def test_deepseek_streaming_round_trip(deepseek_client: AsyncClient, httpx_mock, deepseek_app_config: AppConfig):
    """Verify streaming with reasoning_content accumulation for DeepSeek."""
    sse_chunks = [
        b'data: {"id":"ds-stream","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"Think"}}]}\n\n',
        b'data: {"id":"ds-stream","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"ing step by step"}}]}\n\n',
        b'data: {"id":"ds-stream","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":"The answer"}}]}\n\n',
        b'data: {"id":"ds-stream","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":" is 42."}}]}\n\n',
        b'data: {"id":"ds-stream","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def stream_response(request: httpx.Request):
        content = request.content

        async def body():
            for chunk in sse_chunks:
                yield chunk

        return httpx.Response(200, content=body(), headers={"Content-Type": "text/event-stream"})

    httpx_mock.add_callback(stream_response)

    payload = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Life the universe and everything"}],
        "stream": True,
    }
    resp = await deepseek_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["Content-Type"]

    # Read all streamed chunks
    body_bytes = b""
    async for chunk in resp.aiter_bytes():
        body_bytes += chunk
    body_text = body_bytes.decode()

    # Verify all SSE events were relayed
    assert "Think" in body_text
    assert "ing step by step" in body_text
    assert "The answer" in body_text
    assert "is 42." in body_text
    assert "[DONE]" in body_text

    # Verify reasoning was cached (check the store directly)
    from aslor.reasoning.detector import detect
    detection = detect(payload, deepseek_app_config)
    store = deepseek_client._transport.app.state.store
    cached = store.load(detection.session_key)
    assert cached is not None
    assert "reasoning_content" in cached
    assert cached["reasoning_content"] == "Thinking step by step"


@pytest.mark.asyncio
async def test_deepseek_chat_model_no_repair(deepseek_client: AsyncClient, httpx_mock):
    """deepseek-chat (V3) should NOT trigger reasoning state repair."""
    upstream = {
        "id": "ds-chat-1",
        "object": "chat.completion",
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help?"},
                "finish_reason": "stop",
            }
        ],
    }
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.test/v1/chat/completions",
        status_code=200,
        json=upstream,
    )
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ],
        "stream": False,
    }
    resp = await deepseek_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_deepseek_request_headers_contain_auth(deepseek_client: AsyncClient, httpx_mock):
    """Verify the forwarded request includes the DeepSeek auth header."""
    captured_headers = []

    def capture(request: httpx.Request) -> httpx.Response:
        captured_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "id": "ds-auth",
                "object": "chat.completion",
                "model": "deepseek-reasoner",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    httpx_mock.add_callback(capture)

    payload = {"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "test"}], "stream": False}
    resp = await deepseek_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert len(captured_headers) == 1
    assert captured_headers[0]["authorization"] == "Bearer sk-deepseek-test"
    assert captured_headers[0]["content-type"] == "application/json"


@pytest.mark.asyncio
async def test_auth_passthrough_from_client_header(deepseek_client: AsyncClient, httpx_mock):
    """When Android Studio sends an Authorization header, the proxy uses it upstream."""
    captured_headers = []

    def capture(request: httpx.Request) -> httpx.Response:
        captured_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "id": "ds-passthrough",
                "object": "chat.completion",
                "model": "deepseek-reasoner",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    httpx_mock.add_callback(capture)

    payload = {"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "test"}], "stream": False}
    resp = await deepseek_client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"Authorization": "Bearer sk-client-supplied-key"},
    )
    assert resp.status_code == 200
    assert len(captured_headers) == 1
    # Must use the client's key, NOT the config key (sk-deepseek-test)
    assert captured_headers[0]["authorization"] == "Bearer sk-client-supplied-key"


@pytest.mark.asyncio
async def test_auth_falls_back_to_config_when_no_header(deepseek_client: AsyncClient, httpx_mock):
    """When no Authorization header is sent, the config API key is used."""
    captured_headers = []

    def capture(request: httpx.Request) -> httpx.Response:
        captured_headers.append(dict(request.headers))
        return httpx.Response(
            200,
            json={
                "id": "ds-fallback",
                "object": "chat.completion",
                "model": "deepseek-reasoner",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    httpx_mock.add_callback(capture)

    payload = {"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "test"}], "stream": False}
    resp = await deepseek_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert len(captured_headers) == 1
    assert captured_headers[0]["authorization"] == "Bearer sk-deepseek-test"


@pytest.mark.asyncio
async def test_deepseek_cache_clear_admin(deepseek_client: AsyncClient):
    resp = await deepseek_client.post("/admin/cache/clear")
    assert resp.status_code == 200
    assert resp.json()["deleted"] >= 0


@pytest.mark.asyncio
async def test_default_model_is_applied_when_request_omits_model(
    deepseek_passthrough_client: AsyncClient,
    httpx_mock,
):
    captured_bodies = []

    def capture(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "ds-default-model",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
            },
        )

    httpx_mock.add_callback(capture)

    payload = {"messages": [{"role": "user", "content": "Hello"}], "stream": False}
    resp = await deepseek_passthrough_client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert len(captured_bodies) == 1
    assert captured_bodies[0]["model"] == "deepseek-v4-pro"


@pytest.mark.asyncio
async def test_passthrough_default_deepseek_model_repairs_reasoning_content(
    deepseek_passthrough_client: AsyncClient,
    httpx_mock,
):
    turn1_response = {
        "id": "ds-pass-1",
        "object": "chat.completion",
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "First answer",
                    "reasoning_content": "Cached deepseek reasoning",
                },
                "finish_reason": "stop",
            }
        ],
    }
    httpx_mock.add_response(
        method="POST",
        url="https://api.deepseek.test/v1/chat/completions",
        status_code=200,
        json=turn1_response,
    )

    resp1 = await deepseek_passthrough_client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "First question"}], "stream": False},
    )
    assert resp1.status_code == 200

    captured_bodies = []

    def capture_turn2(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "ds-pass-2",
                "object": "chat.completion",
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Second answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(capture_turn2)

    resp2 = await deepseek_passthrough_client.post(
        "/v1/chat/completions",
        json={
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Follow up question"},
            ],
            "stream": False,
        },
    )
    assert resp2.status_code == 200
    assert len(captured_bodies) == 1

    forwarded_body = captured_bodies[0]
    assert forwarded_body["model"] == "deepseek-v4-pro"
    assistant_msg = next(m for m in forwarded_body["messages"] if m["role"] == "assistant")
    assert assistant_msg["reasoning_content"] == "Cached deepseek reasoning"
