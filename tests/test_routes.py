"""Integration tests for the FastAPI routes.

Uses httpx.AsyncClient with pytest-asyncio.  Upstream HTTP calls are mocked
using pytest-httpx so no real network requests are made.
"""

from __future__ import annotations

import json
import os
import pytest
import pytest_asyncio
import httpx
import yaml
from httpx import AsyncClient

from aslor.config import AppConfig
from aslor.server.app import create_app
from aslor.server import routes as routes_module
from aslor.missions.evaluator import evaluate_response
from aslor.missions.injector import inject_mission_context
from aslor.missions.models import MissionState


@pytest.fixture
def app_config(tmp_path):
    os.environ["TEST_KEY"] = "sk-test"
    return AppConfig(
        {
            "server": {"host": "127.0.0.1", "port": 8080},
            "provider": {
                "name": "openai",
                "base_url": "https://api.openai.test/v1",
                "api_key_env": "TEST_KEY",
                "timeout_seconds": 10,
            },
            "cache": {"path": str(tmp_path / "test.db"), "encrypt": False},
            "logging": {"level": "WARNING", "format": "text"},
        }
    )


@pytest_asyncio.fixture
async def client(app_config: AppConfig):
    app = create_app(app_config)
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_status_endpoint(client: AsyncClient):
    resp = await client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "running"
    assert "uptime_seconds" in data
    assert data["provider"] == "openai"


@pytest.mark.asyncio
async def test_clear_cache(client: AsyncClient):
    resp = await client.post("/admin/cache/clear")
    assert resp.status_code == 200
    assert "deleted" in resp.json()


@pytest.mark.asyncio
async def test_provider_catalog_endpoint(client: AsyncClient):
    resp = await client.get("/admin/provider-catalog")
    assert resp.status_code == 200
    data = resp.json()
    assert "providers" in data
    assert any(p["name"] == "openai" for p in data["providers"])


@pytest.mark.asyncio
async def test_restart_endpoint_schedules_restart(client: AsyncClient, monkeypatch):
    called = {"value": False}

    def fake_schedule_restart():
        called["value"] = True

    monkeypatch.setattr(routes_module, "_schedule_restart", fake_schedule_restart)

    resp = await client.post("/admin/restart")
    assert resp.status_code == 202
    assert resp.json()["restarting"] is True
    assert called["value"] is True


@pytest.mark.asyncio
async def test_update_config_passthrough_saves_minimal_provider_fields(client: AsyncClient, tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("ASLOR_CONFIG", str(config_path))

    payload = {
        "provider": {
            "name": "passthrough",
            "base_url": "https://api.deepseek.com/v1",
        },
        "server": {"host": "127.0.0.1", "port": 3001},
    }

    resp = await client.put("/admin/config", json=payload)
    assert resp.status_code == 200

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["provider"] == {
        "name": "passthrough",
        "base_url": "https://api.deepseek.com/v1",
    }


@pytest.mark.asyncio
async def test_chat_completions_bad_json(client: AsyncClient):
    resp = await client.post("/v1/chat/completions", content=b"not json", headers={"Content-Type": "application/json"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_chat_completions_upstream_error(client: AsyncClient, httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        status_code=400,
        json={"error": {"message": "The reasoning_content in the thinking mode must be passed back."}},
    )
    payload = {
        "model": "o1-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "Upstream error" in data["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_missing_reasoning_error_retries_with_stripped_history(client: AsyncClient, httpx_mock):
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        has_assistant = any(m.get("role") == "assistant" for m in body.get("messages", []))
        if has_assistant:
            return httpx.Response(
                400,
                json={"error": {"message": "The reasoning_content in the thinking mode must be passed back to the API."}},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-retry-ok",
                "object": "chat.completion",
                "model": body.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok", "reasoning_content": "r"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(
        handler,
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        is_reusable=True,
    )

    payload = {
        "model": "o1-mini",
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Turn 1 answer"},
            {"role": "user", "content": "Follow up"},
        ],
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert len(captured) == 2
    assert any(m.get("role") == "assistant" for m in captured[0].get("messages", []))
    assert not any(m.get("role") == "assistant" for m in captured[1].get("messages", []))


@pytest.mark.asyncio
async def test_missing_reasoning_error_retries_with_thinking_disabled(client: AsyncClient, httpx_mock):
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(
                400,
                json={"error": {"message": "The reasoning_content in the thinking mode must be passed back to the API."}},
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-retry2-ok",
                "object": "chat.completion",
                "model": body.get("model", ""),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok", "reasoning_content": "r"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(
        handler,
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        is_reusable=True,
    )

    payload = {
        "model": "o1-mini",
        "reasoning_effort": "high",
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Turn 1 answer"},
            {"role": "user", "content": "Follow up"},
        ],
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert len(captured) == 3
    assert any(m.get("role") == "assistant" for m in captured[0].get("messages", []))
    assert not any(m.get("role") == "assistant" for m in captured[1].get("messages", []))
    assert "reasoning_effort" in captured[1]
    assert "reasoning_effort" not in captured[2]


@pytest.mark.asyncio
async def test_context_length_error_retries_with_reduced_history(client: AsyncClient, httpx_mock):
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        messages = body.get("messages", [])
        if isinstance(messages, list) and len(json.dumps(messages)) > 3600:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "This model's maximum context length is 1000 tokens. However, you requested 4047393 tokens (4047393 in the messages, 0 in the completion).",
                        "type": "invalid_request_error",
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-context-ok",
                "object": "chat.completion",
                "model": body.get("model", ""),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )

    httpx_mock.add_callback(
        handler,
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        is_reusable=True,
    )

    payload = {
        "model": "o1-mini",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a" * 5000, "reasoning_content": "r1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "b" * 5000, "reasoning_content": "r2"},
            {"role": "user", "content": "q3"},
        ],
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert len(captured) >= 2
    assert len(captured[0]["messages"]) == 6
    assert len(json.dumps(captured[-1]["messages"])) <= 3600


@pytest.mark.asyncio
async def test_unknown_parameter_is_stripped_and_retried(client: AsyncClient, httpx_mock):
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "Unknown parameter: reasoning_effort"}})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-ok",
                "object": "chat.completion",
                "model": body.get("model", ""),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )

    httpx_mock.add_callback(
        handler,
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        is_reusable=True,
    )

    payload = {
        "model": "o1-mini",
        "reasoning_effort": "high",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"
    assert "reasoning_effort" in captured[0]
    assert "reasoning_effort" not in captured[-1]


@pytest.mark.asyncio
async def test_rate_limit_is_retried_then_returns_message_on_failure(client: AsyncClient, httpx_mock):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 2:
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-ok2",
                "object": "chat.completion",
                "model": "o1-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            },
        )

    httpx_mock.add_callback(
        handler,
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        is_reusable=True,
    )

    payload = {
        "model": "o1-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_streaming_error_returns_valid_sse_chunks(client: AsyncClient, httpx_mock):
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        status_code=400,
        json={"error": {"message": "bad request"}},
    )
    payload = {
        "model": "o1-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("Content-Type", "")
    body = (await resp.aread()).decode("utf-8", errors="replace")
    assert "chat.completion.chunk" in body
    assert "data: [DONE]" in body


def test_mission_attempts_do_not_exceed_max_attempts_without_terminal_signal():
    mission = MissionState(
        mission_id="m",
        title="t",
        description="d",
        success_criteria="something",
        max_attempts=3,
        status="active",
        current_attempt=2,
        created_at=0,
        updated_at=0,
    )
    result = evaluate_response({"choices": [{"message": {"role": "assistant", "content": "working"}}]}, mission)
    assert result.failed is True
    assert result.attempt == 3


def test_mission_injector_clamps_attempt_display():
    mission = MissionState(
        mission_id="m",
        title="t",
        description="d",
        success_criteria="s",
        max_attempts=3,
        status="active",
        current_attempt=60,
        created_at=0,
        updated_at=0,
    )
    out = inject_mission_context([{"role": "user", "content": "hi"}], mission, skills=[])
    assert out[0]["role"] == "system"
    assert "Attempt 3 of 3" in out[0]["content"]


@pytest.mark.asyncio
async def test_chat_completions_success_non_streaming(client: AsyncClient, httpx_mock):
    upstream_response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "o1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Here is my answer.",
                    "reasoning_content": "I thought carefully.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        status_code=200,
        json=upstream_response,
    )
    payload = {
        "model": "o1-mini",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Here is my answer."


@pytest.mark.asyncio
async def test_reasoning_state_is_cached_and_repaired(client: AsyncClient, httpx_mock, app_config: AppConfig):
    """Full round-trip: first turn caches reasoning state; second turn injects it."""
    first_response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "o1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Turn 1 answer",
                    "reasoning_content": "Cached reasoning from turn 1",
                },
                "finish_reason": "stop",
            }
        ],
    }
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        status_code=200,
        json=first_response,
    )

    turn1_payload = {
        "model": "o1-mini",
        "messages": [{"role": "user", "content": "First question"}],
        "stream": False,
    }
    resp1 = await client.post("/v1/chat/completions", json=turn1_payload)
    assert resp1.status_code == 200

    captured_bodies = []

    def capture_request(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "model": "o1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Turn 2 answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(capture_request)

    turn2_payload = {
        "model": "o1-mini",
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "Turn 1 answer"},
            {"role": "user", "content": "Follow up question"},
        ],
        "stream": False,
    }
    resp2 = await client.post("/v1/chat/completions", json=turn2_payload)
    assert resp2.status_code == 200

    assert len(captured_bodies) == 1
    forwarded_messages = captured_bodies[0]["messages"]
    assistant_msg = next(m for m in forwarded_messages if m["role"] == "assistant")
    assert assistant_msg.get("reasoning_content") == "Cached reasoning from turn 1"


@pytest.mark.asyncio
async def test_reasoning_history_repairs_multiple_prior_assistant_messages(client: AsyncClient, httpx_mock):
    turn1_response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "o1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Turn 1 answer",
                    "reasoning_content": "Reasoning 1",
                },
                "finish_reason": "stop",
            }
        ],
    }
    turn2_response = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "o1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Turn 2 answer",
                    "reasoning_content": "Reasoning 2",
                },
                "finish_reason": "stop",
            }
        ],
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        status_code=200,
        json=turn1_response,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        status_code=200,
        json=turn2_response,
    )

    resp1 = await client.post(
        "/v1/chat/completions",
        json={"model": "o1-mini", "messages": [{"role": "user", "content": "First question"}], "stream": False},
    )
    assert resp1.status_code == 200

    resp2 = await client.post(
        "/v1/chat/completions",
        json={
            "model": "o1-mini",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Turn 1 answer"},
                {"role": "user", "content": "Second question"},
            ],
            "stream": False,
        },
    )
    assert resp2.status_code == 200

    captured_bodies = []

    def capture_turn3(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_bodies.append(body)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-3",
                "object": "chat.completion",
                "model": "o1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Turn 3 answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(capture_turn3)

    resp3 = await client.post(
        "/v1/chat/completions",
        json={
            "model": "o1-mini",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Turn 1 answer"},
                {"role": "user", "content": "Second question"},
                {"role": "assistant", "content": "Turn 2 answer"},
                {"role": "user", "content": "Third question"},
            ],
            "stream": False,
        },
    )
    assert resp3.status_code == 200

    assert len(captured_bodies) == 1
    assistant_messages = [m for m in captured_bodies[0]["messages"] if m["role"] == "assistant"]
    assert assistant_messages[0]["reasoning_content"] == "Reasoning 1"
    assert assistant_messages[1]["reasoning_content"] == "Reasoning 2"


@pytest.mark.asyncio
async def test_reasoning_history_repairs_drifted_assistant_replays_by_order(client: AsyncClient, httpx_mock):
    turn1_response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "o1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Turn 1 answer",
                    "reasoning_content": "Reasoning 1",
                },
                "finish_reason": "stop",
            }
        ],
    }
    turn2_response = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "model": "o1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Turn 2 answer",
                    "reasoning_content": "Reasoning 2",
                },
                "finish_reason": "stop",
            }
        ],
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        status_code=200,
        json=turn1_response,
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        status_code=200,
        json=turn2_response,
    )

    await client.post(
        "/v1/chat/completions",
        json={"model": "o1-mini", "messages": [{"role": "user", "content": "First question"}], "stream": False},
    )
    await client.post(
        "/v1/chat/completions",
        json={
            "model": "o1-mini",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Turn 1 answer"},
                {"role": "user", "content": "Second question"},
            ],
            "stream": False,
        },
    )

    captured_bodies = []

    def capture_turn3(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-3",
                "object": "chat.completion",
                "model": "o1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Turn 3 answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(capture_turn3)

    resp3 = await client.post(
        "/v1/chat/completions",
        json={
            "model": "o1-mini",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Turn 1 answer\n"},
                {"role": "user", "content": "Second question"},
                {"role": "assistant", "content": "Turn 2 answer (replayed)"},
                {"role": "user", "content": "Third question"},
            ],
            "stream": False,
        },
    )
    assert resp3.status_code == 200

    assistant_messages = [m for m in captured_bodies[0]["messages"] if m["role"] == "assistant"]
    assert assistant_messages[0]["reasoning_content"] == "Reasoning 1"
    assert assistant_messages[1]["reasoning_content"] == "Reasoning 2"


@pytest.mark.asyncio
async def test_reasoning_repair_overrides_incorrect_existing_reasoning_content(client: AsyncClient, httpx_mock):
    first_response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "o1-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Turn 1 answer",
                    "reasoning_content": "Correct reasoning",
                },
                "finish_reason": "stop",
            }
        ],
    }
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.test/v1/chat/completions",
        status_code=200,
        json=first_response,
    )

    resp1 = await client.post(
        "/v1/chat/completions",
        json={"model": "o1-mini", "messages": [{"role": "user", "content": "First question"}], "stream": False},
    )
    assert resp1.status_code == 200

    captured_bodies = []

    def capture_turn2(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "model": "o1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Turn 2 answer"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(capture_turn2)

    resp2 = await client.post(
        "/v1/chat/completions",
        json={
            "model": "o1-mini",
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Turn 1 answer", "reasoning_content": "Wrong reasoning"},
                {"role": "user", "content": "Follow up"},
            ],
            "stream": False,
        },
    )
    assert resp2.status_code == 200
    assistant_msg = next(m for m in captured_bodies[0]["messages"] if m["role"] == "assistant")
    assert assistant_msg["reasoning_content"] == "Correct reasoning"
