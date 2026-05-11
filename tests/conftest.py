"""Shared pytest fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aslor.cache.db import CacheDB
from aslor.config import AppConfig
from aslor.reasoning.state import ReasoningStateStore


@pytest.fixture
def tmp_db(tmp_path: Path) -> CacheDB:
    db = CacheDB(tmp_path / "test.db", encrypt=False)
    yield db
    db.close()


@pytest.fixture
def tmp_db_encrypted(tmp_path: Path) -> CacheDB:
    db = CacheDB(tmp_path / "test_enc.db", encrypt=True)
    yield db
    db.close()


@pytest.fixture
def store(tmp_db: CacheDB) -> ReasoningStateStore:
    return ReasoningStateStore(tmp_db)


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(
        {
            "server": {"host": "127.0.0.1", "port": 8080},
            "provider": {
                "name": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key_env": "TEST_API_KEY",
            },
            "cache": {"path": ":memory:", "encrypt": False},
        }
    )
