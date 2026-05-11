"""FastAPI route handlers."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import base64
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from aslor.config import AppConfig
from aslor.missions.models import MissionState as MissionModel
from aslor.missions.registry import SkillRegistry
from aslor.missions.state import MissionStateStore
from aslor.models.registry import get_registry
from aslor.pipeline import RequestPipeline
from aslor.reasoning.state import ReasoningStateStore
from aslor.server.log_buffer import push as log_push, snapshot as log_snapshot
from aslor.server.stats import StatsTracker
from aslor.vision.store import VisionStore

logger = logging.getLogger(__name__)

_DASHBOARD_HTML: str | None = None


def _load_dashboard_html() -> str:
    global _DASHBOARD_HTML
    if _DASHBOARD_HTML is None:
        html_path = Path(__file__).resolve().parent / "dashboard.html"
        if html_path.exists():
            _DASHBOARD_HTML = html_path.read_text(encoding="utf-8")
        else:
            _DASHBOARD_HTML = "<html><body><h1>Dashboard not found</h1></body></html>"
    return _DASHBOARD_HTML


def _extract_api_key(request: Request, config: AppConfig) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return config.provider.api_key


def _extract_session_hint(request: Request) -> str:
    candidates = [
        request.headers.get("X-Conversation-Id", ""),
        request.headers.get("X-Session-Id", ""),
        request.headers.get("X-Thread-Id", ""),
        request.headers.get("OpenAI-Conversation-ID", ""),
        request.headers.get("OpenAI-Thread-ID", ""),
        request.headers.get("X-Request-Id", ""),
        request.headers.get("X-Correlation-Id", ""),
    ]
    for c in candidates:
        s = str(c or "").strip()
        if len(s) >= 6:
            return s
    return ""


def _config_path() -> Path:
    return Path(os.environ.get("ASLOR_CONFIG", "config.yaml"))


def _read_raw_config() -> dict[str, Any]:
    path = _config_path()
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _write_raw_config(raw: dict[str, Any]) -> None:
    path = _config_path()
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(raw, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)


def _schedule_restart(restart_delay_seconds: float = 0.5) -> None:
    """Restart ASLOR without creating duplicate listeners on the same port."""
    reload_enabled = os.environ.get("ASLOR_RELOAD", "").strip().lower() in {"1", "true", "yes", "on"}
    if reload_enabled:
        target = _config_path()
        if not target.exists():
            target = Path(__file__)
        os.utime(target, None)
        return

    def _restart_current_process() -> None:
        time.sleep(max(restart_delay_seconds, 0.1))
        os.execve(sys.executable, [sys.executable, "-m", "aslor.main"], os.environ.copy())

    threading.Thread(target=_restart_current_process, daemon=True).start()


def build_router(
    config: AppConfig,
    store: ReasoningStateStore,
    pipeline: RequestPipeline,
    stats: StatsTracker | None = None,
    mission_store: MissionStateStore | None = None,
    skill_registry: SkillRegistry | None = None,
    vision_store: VisionStore | None = None,
) -> APIRouter:
    router = APIRouter()
    _start_time = time.time()

    # ── Dashboard ──────────────────────────────────────────────────────────

    @router.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        return _load_dashboard_html()

    # ── Config CRUD ────────────────────────────────────────────────────────

    @router.get("/admin/config")
    async def get_config() -> JSONResponse:
        raw = _read_raw_config()
        return JSONResponse(raw)

    @router.put("/admin/config")
    async def update_config(request: Request) -> JSONResponse:
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        allowed = ("server", "provider", "cache", "logging", "missions", "vision")
        sanitized: dict[str, Any] = {k: body[k] for k in allowed if k in body}

        if not sanitized:
            raise HTTPException(status_code=400, detail="No valid config sections provided")

        _write_raw_config(sanitized)
        logger.info("admin: config written to %s — restart required", _config_path())
        return JSONResponse({"saved": True, "path": str(_config_path())})

    # ── Provider presets ───────────────────────────────────────────────────

    @router.post("/admin/restart")
    async def restart_proxy(background_tasks: BackgroundTasks) -> JSONResponse:
        logger.info("admin: restart requested")
        background_tasks.add_task(_schedule_restart)
        return JSONResponse({"restarting": True}, status_code=202)

    @router.get("/admin/presets")
    async def get_presets() -> JSONResponse:
        return JSONResponse(get_registry().get_all_presets())

    @router.get("/admin/provider-catalog")
    async def get_provider_catalog() -> JSONResponse:
        reg = get_registry()
        return JSONResponse(
            {
                "providers": [p.to_catalog_dict() for p in reg.list_providers()],
            }
        )

    # ── API key status ─────────────────────────────────────────────────────

    @router.get("/admin/key-status")
    async def key_status(request: Request) -> JSONResponse:
        api_key = _extract_api_key(request, config)
        env_key = config.provider.api_key
        header_key = ""
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            header_key = auth[7:].strip()

        return JSONResponse({
            "available": bool(api_key),
            "source": "header" if header_key else ("env" if env_key else "none"),
            "masked": _mask_key(api_key),
            "env_var": config.provider.api_key_env,
        })

    # ── Connection test ────────────────────────────────────────────────────

    @router.post("/admin/connection-test")
    async def connection_test(request: Request) -> JSONResponse:
        from aslor.providers.registry import get_adapter

        api_key = _extract_api_key(request, config)
        if not api_key:
            raise HTTPException(status_code=400, detail="No API key configured")

        adapter = get_adapter(config)
        url = f"{adapter.get_base_url()}/models"
        headers = adapter.get_headers(api_key)
        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, headers=headers)
            elapsed = round((time.time() - t0) * 1000)
            body: Any = None
            try:
                body = resp.json()
            except Exception:
                body = resp.text[:500] if resp.text else None
            return JSONResponse({
                "ok": resp.status_code < 400,
                "status_code": resp.status_code,
                "latency_ms": elapsed,
                "body": body,
            })
        except httpx.RequestError as exc:
            elapsed = round((time.time() - t0) * 1000)
            return JSONResponse({
                "ok": False,
                "status_code": 0,
                "latency_ms": elapsed,
                "error": str(exc),
            })

    # ── Request stats ──────────────────────────────────────────────────────

    @router.get("/admin/stats")
    async def get_stats() -> JSONResponse:
        if stats is None:
            return JSONResponse({"error": "Stats not enabled"})
        snap = stats.snapshot()
        return JSONResponse({
            "total_requests": snap.total_requests,
            "success_count": snap.success_count,
            "error_count": snap.error_count,
            "avg_latency_ms": snap.avg_latency_ms,
            "repair_count": snap.repair_count,
            "recent_requests": snap.recent_requests[:50],
            "uptime_seconds": snap.uptime_seconds,
        })

    # ── Chat completions ───────────────────────────────────────────────────

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> StreamingResponse:
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        api_key = _extract_api_key(request, config)
        session_hint = _extract_session_hint(request)
        return await pipeline.run(body, api_key=api_key, session_hint=session_hint or None)

    @router.post("/chat/completions")
    async def chat_completions_compat(request: Request) -> StreamingResponse:
        return await chat_completions(request)

    # ── Models ─────────────────────────────────────────────────────────────

    @router.get("/v1/models")
    async def list_models(request: Request) -> JSONResponse:
        # Derive model catalog from the data-driven registry (models.yaml).
        provider = config.provider.name
        reg = get_registry()
        entries = reg.list_models(provider)
        if not entries:
            entries = reg.list_models()  # fallback: all known models

        models = [e.to_model_dict() for e in entries]

        # Optionally try to enrich with live upstream data if a key is available.
        api_key = _extract_api_key(request, config)
        if api_key:
            try:
                from aslor.providers.registry import get_adapter
                adapter = get_adapter(config)
                url = f"{adapter.get_base_url()}/models"
                headers = adapter.get_headers(api_key)
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(url, headers=headers)
                if resp.status_code < 400:
                    try:
                        live = resp.json()
                        if isinstance(live, dict) and "data" in live:
                            models = live["data"]
                    except (json.JSONDecodeError, ValueError):
                        pass
            except Exception:
                pass  # fall back to static catalog

        return JSONResponse({"object": "list", "data": models})

    @router.get("/models")
    async def list_models_compat(request: Request) -> JSONResponse:
        return await list_models(request)

    @router.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy_v1(request: Request, path: str) -> Response:
        api_key = _extract_api_key(request, config)
        from aslor.providers.registry import get_adapter

        adapter = get_adapter(config)
        upstream_url = f"{adapter.get_base_url()}/{path.lstrip('/')}"
        upstream_headers = dict(request.headers)
        upstream_headers.pop("host", None)
        upstream_headers.pop("content-length", None)
        upstream_headers.update(adapter.get_headers(api_key))

        body_bytes = await request.body()
        timeout = config.provider.timeout_seconds
        client = httpx.AsyncClient(timeout=timeout)
        resp = await client.send(
            client.build_request(
                request.method,
                upstream_url,
                params=request.query_params,
                headers=upstream_headers,
                content=body_bytes if body_bytes else None,
            ),
            stream=True,
        )

        response_headers = {}
        for k, v in resp.headers.items():
            lk = k.lower()
            if lk in {"content-length", "transfer-encoding", "connection", "content-encoding"}:
                continue
            response_headers[k] = v

        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" in content_type.lower():
            async def _stream():
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()
                    await client.aclose()

            return StreamingResponse(
                content=_stream(),
                status_code=resp.status_code,
                headers=response_headers,
                media_type=content_type,
            )

        data = await resp.aread()
        await resp.aclose()
        await client.aclose()
        return Response(content=data, status_code=resp.status_code, headers=response_headers, media_type=content_type)

    # ── Status ─────────────────────────────────────────────────────────────

    @router.get("/status")
    async def status() -> JSONResponse:
        uptime_seconds = round(time.time() - _start_time)
        return JSONResponse(
            {
                "status": "running",
                "uptime_seconds": uptime_seconds,
                "provider": config.provider.name,
                "provider_base_url": config.provider.base_url,
                "cache_entries": store.count(),
                "cache_encrypted": config.cache.encrypt,
            }
        )

    # ── Cache management ───────────────────────────────────────────────────

    @router.get("/admin/cache/entries")
    async def list_cache_entries() -> JSONResponse:
        entries = store.list_entries()
        return JSONResponse(entries)

    @router.delete("/admin/cache/entries/{session_id}")
    async def delete_cache_entry(session_id: str) -> JSONResponse:
        store.delete(session_id)
        logger.info("admin: deleted cache entry session=%s", session_id)
        return JSONResponse({"deleted": session_id})

    @router.post("/admin/cache/clear")
    async def clear_cache() -> JSONResponse:
        deleted = store.clear_all()
        logger.info("admin: cache cleared (%d entries removed)", deleted)
        return JSONResponse({"deleted": deleted})

    # â”€â”€ Vision â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @router.post("/admin/vision/images")
    async def upload_vision_image(request: Request) -> JSONResponse:
        if vision_store is None or not config.vision.enabled:
            raise HTTPException(status_code=400, detail="Vision sidecar not enabled")

        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        content_base64 = str(body.get("content_base64") or "")
        if not content_base64:
            raise HTTPException(status_code=400, detail="content_base64 is required")

        try:
            content = base64.b64decode(content_base64, validate=True)
        except Exception:
            raise HTTPException(status_code=400, detail="content_base64 must be valid base64")

        if len(content) > config.vision.max_image_bytes:
            raise HTTPException(status_code=413, detail="Image exceeds configured size limit")

        meta = vision_store.save_image(
            content=content,
            filename=str(body.get("filename") or ""),
            mime_type=str(body.get("mime_type") or ""),
        )
        return JSONResponse(
            {
                "image_id": meta["image_id"],
                "mime_type": meta["mime_type"],
                "size_bytes": meta["size_bytes"],
                "image_url": f"aslor://image/{meta['image_id']}",
            },
            status_code=201,
        )

    # ── Missions ────────────────────────────────────────────────────────────

    @router.get("/admin/missions")
    async def list_missions() -> JSONResponse:
        if mission_store is None:
            return JSONResponse({"missions": [], "error": "Missions not enabled"})
        missions = mission_store.list_all()
        return JSONResponse({"missions": [m.to_dict() for m in missions]})

    @router.post("/admin/missions")
    async def create_mission(request: Request) -> JSONResponse:
        if mission_store is None:
            raise HTTPException(status_code=400, detail="Missions not enabled")
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        mission_id = body.get("mission_id", "").strip()
        if not mission_id:
            raise HTTPException(status_code=400, detail="mission_id is required")

        existing = mission_store.load(mission_id)
        if existing is not None:
            raise HTTPException(
                status_code=409,
                detail=f"Mission '{mission_id}' already exists",
            )

        # Auto-pause any other active mission
        active = mission_store.get_active_mission()
        if active is not None and body.get("status", "active") == "active":
            active.status = "paused"
            mission_store.save(active)

        mission = MissionModel(
            mission_id=mission_id,
            title=body.get("title", mission_id),
            description=body.get("description", ""),
            success_criteria=body.get("success_criteria", ""),
            max_attempts=int(body.get("max_attempts", 3)),
            status=body.get("status", "active"),
            current_attempt=0,
            created_at=int(time.time()),
            updated_at=int(time.time()),
            completed_at=None,
            notes="",
            skill_ids=body.get("skill_ids"),
        )
        mission_store.create(mission)
        logger.info("admin: mission '%s' created", mission_id)
        return JSONResponse(mission.to_dict(), status_code=201)

    @router.get("/admin/missions/{mission_id}")
    async def get_mission(mission_id: str) -> JSONResponse:
        if mission_store is None:
            raise HTTPException(status_code=400, detail="Missions not enabled")
        mission = mission_store.load(mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="Mission not found")
        return JSONResponse(mission.to_dict())

    @router.put("/admin/missions/{mission_id}")
    async def update_mission(mission_id: str, request: Request) -> JSONResponse:
        if mission_store is None:
            raise HTTPException(status_code=400, detail="Missions not enabled")
        mission = mission_store.load(mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="Mission not found")
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")

        for field in (
            "title", "description", "success_criteria", "status",
            "notes", "skill_ids",
        ):
            if field in body:
                setattr(mission, field, body[field])
        if "max_attempts" in body:
            mission.max_attempts = int(body["max_attempts"])

        # Re-activating resets the attempt counter
        if body.get("status") == "active":
            mission.current_attempt = 0
            mission.completed_at = None

        mission_store.save(mission)
        logger.info("admin: mission '%s' updated", mission_id)
        return JSONResponse(mission.to_dict())

    @router.delete("/admin/missions/{mission_id}")
    async def delete_mission(mission_id: str) -> JSONResponse:
        if mission_store is None:
            raise HTTPException(status_code=400, detail="Missions not enabled")
        mission_store.delete(mission_id)
        logger.info("admin: mission '%s' deleted", mission_id)
        return JSONResponse({"deleted": mission_id})

    @router.post("/admin/missions/{mission_id}/reset")
    async def reset_mission(mission_id: str) -> JSONResponse:
        if mission_store is None:
            raise HTTPException(status_code=400, detail="Missions not enabled")
        mission = mission_store.reset(mission_id)
        if mission is None:
            raise HTTPException(status_code=404, detail="Mission not found")
        logger.info("admin: mission '%s' reset (attempts=%d)", mission_id, mission.current_attempt)
        return JSONResponse(mission.to_dict())

    # ── Skills ──────────────────────────────────────────────────────────────

    @router.get("/admin/skills")
    async def list_skills() -> JSONResponse:
        if skill_registry is None:
            return JSONResponse({"skills": [], "error": "Skills not enabled"})
        return JSONResponse({
            "skills": [s.to_dict() for s in skill_registry.list_skills()],
        })

    @router.put("/admin/skills/{skill_id}")
    async def toggle_skill(skill_id: str, request: Request) -> JSONResponse:
        if skill_registry is None:
            raise HTTPException(status_code=400, detail="Skills not enabled")
        try:
            body: dict[str, Any] = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON body")
        enabled = body.get("enabled")
        if enabled is None:
            raise HTTPException(status_code=400, detail="'enabled' field is required")
        skill_registry.enable_skill(skill_id, enabled)
        skill = skill_registry.get_skill(skill_id)
        if skill is None:
            raise HTTPException(status_code=404, detail="Skill not found")
        return JSONResponse(skill.to_dict())

    # ── Logs ───────────────────────────────────────────────────────────────

    @router.get("/admin/logs")
    async def get_logs() -> JSONResponse:
        return JSONResponse(log_snapshot())

    return router


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]
