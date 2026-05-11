"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
import httpx

from aslor.cache.db import CacheDB
from aslor.config import AppConfig
from aslor.logging_config import configure_logging
from aslor.missions.registry import SkillRegistry, get_skill_registry
from aslor.missions.state import MissionStateStore
from aslor.pipeline import RequestPipeline
from aslor.reasoning.state import ReasoningStateStore
from aslor.server.middleware import RequestLoggingMiddleware
from aslor.server.routes import build_router
from aslor.server.stats import StatsTracker
from aslor.vision.analyzer import VisionAnalyzer
from aslor.vision.store import VisionStore


def create_app(config: AppConfig | None = None) -> FastAPI:
    if config is None:
        config = AppConfig.from_env()

    configure_logging(config.logging.level, config.logging.format)

    db = CacheDB(config.cache.path, encrypt=config.cache.encrypt)
    store = ReasoningStateStore(db)
    vision_store = VisionStore(db, config.vision.upload_dir)
    vision_analyzer = VisionAnalyzer(config.vision, vision_store)
    stats = StatsTracker()

    mission_store = MissionStateStore(db)
    skill_registry: SkillRegistry | None = None
    if config.missions.enabled:
        skill_registry = SkillRegistry()
        skill_registry.load(config.missions.skills_path)

    upstream_client = httpx.AsyncClient(timeout=config.provider.timeout_seconds)
    pipeline = RequestPipeline(
        config,
        store,
        stats=stats,
        mission_store=mission_store,
        skill_registry=skill_registry,
        vision_analyzer=vision_analyzer,
        upstream_client=upstream_client,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await upstream_client.aclose()
        db.close()

    app = FastAPI(
        title="android-studio-llm-openai-reasoning-proxy",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(RequestLoggingMiddleware)

    router = build_router(
        config, store, pipeline, stats=stats,
        mission_store=mission_store,
        skill_registry=skill_registry,
        vision_store=vision_store,
    )
    app.include_router(router)

    app.state.config = config
    app.state.store = store
    app.state.vision_store = vision_store

    return app
