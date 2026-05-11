from __future__ import annotations

import base64
import json
import os

import httpx
import pytest
import pytest_asyncio
from httpx import AsyncClient

from aslor.config import AppConfig
from aslor.server.app import create_app


@pytest.fixture
def vision_app_config(tmp_path):
    os.environ["TEST_KEY"] = "sk-test"
    os.environ["VISION_TEST_KEY"] = "sk-vision-test"
    return AppConfig(
        {
            "server": {"host": "127.0.0.1", "port": 8080},
            "provider": {
                "name": "openai",
                "base_url": "https://api.openai.test/v1",
                "api_key_env": "TEST_KEY",
                "timeout_seconds": 10,
            },
            "cache": {"path": str(tmp_path / "vision_test.db"), "encrypt": False},
            "logging": {"level": "WARNING", "format": "text"},
            "vision": {
                "enabled": True,
                "base_url": "https://vision.test/v1",
                "model": "gpt-4o-mini",
                "api_key_env": "VISION_TEST_KEY",
                "timeout_seconds": 10,
                "upload_dir": str(tmp_path / "uploads"),
                "max_image_bytes": 1024 * 1024,
            },
        }
    )


@pytest_asyncio.fixture
async def vision_client(vision_app_config: AppConfig):
    app = create_app(vision_app_config)
    transport = httpx.ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_upload_vision_image(vision_client: AsyncClient):
    png_bytes = b"\x89PNG\r\n\x1a\nfakepng"
    resp = await vision_client.post(
        "/admin/vision/images",
        json={
            "filename": "screen.png",
            "mime_type": "image/png",
            "content_base64": base64.b64encode(png_bytes).decode("ascii"),
        },
    )

    assert resp.status_code == 201
    data = resp.json()
    assert data["image_url"].startswith("aslor://image/")
    assert data["mime_type"] == "image/png"


@pytest.mark.asyncio
async def test_chat_completion_injects_visual_context(vision_client: AsyncClient, httpx_mock):
    upload_resp = await vision_client.post(
        "/admin/vision/images",
        json={
            "filename": "screen.png",
            "mime_type": "image/png",
            "content_base64": base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode("ascii"),
        },
    )
    assert upload_resp.status_code == 201
    image_url = upload_resp.json()["image_url"]

    vision_payloads: list[dict] = []
    main_payloads: list[dict] = []

    def capture_vision(request: httpx.Request) -> httpx.Response:
        vision_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "screen_type": "android_settings",
                                    "summary": "A settings screen with a clipped save button.",
                                    "visible_text": ["Settings", "Notifications", "Save"],
                                    "layout_issues": [{"severity": "high", "issue": "Save button is clipped"}],
                                    "accessibility_issues": [{"severity": "medium", "issue": "Low contrast subtitle"}],
                                    "component_tree": [],
                                }
                            )
                        }
                    }
                ]
            },
        )

    def capture_main(request: httpx.Request) -> httpx.Response:
        main_payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-vision",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "I see the issue."},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    httpx_mock.add_callback(capture_vision, method="POST", url="https://vision.test/v1/chat/completions")
    httpx_mock.add_callback(capture_main, method="POST", url="https://api.openai.test/v1/chat/completions")

    resp = await vision_client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Debug this Android screen"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "stream": False,
        },
    )

    assert resp.status_code == 200
    assert len(vision_payloads) == 1
    vision_content = vision_payloads[0]["messages"][1]["content"]
    assert any(part["type"] == "image_url" for part in vision_content)

    assert len(main_payloads) == 1
    forwarded_messages = main_payloads[0]["messages"]
    assert forwarded_messages[0]["role"] == "system"
    assert "VISUAL_CONTEXT:" in forwarded_messages[0]["content"]
    assert "Save button is clipped" in forwarded_messages[0]["content"]
    assert forwarded_messages[1]["content"] == "Debug this Android screen\n[image]"
