"""Config loader — reads config.yaml and expands env vars."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml


class ServerConfig:
    def __init__(self, raw: dict) -> None:
        self.host: str = raw.get("host", "127.0.0.1")
        self.port: int = int(raw.get("port", 3001))


class ProviderConfig:
    def __init__(self, raw: dict) -> None:
        self.name: str = raw.get("name", "openai")
        self.base_url: str = raw.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.api_key_env: str = raw.get("api_key_env", "OPENAI_API_KEY")
        self.default_model: str = raw.get("default_model", "")
        self.timeout_seconds: int = int(raw.get("timeout_seconds", 120))
        self.thinking_enabled: bool = bool(raw.get("thinking_enabled", False))
        self.thinking_budget_tokens: int = int(raw.get("thinking_budget_tokens", 5000))

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        return key


class CacheConfig:
    def __init__(self, raw: dict) -> None:
        self.path: str = raw.get("path", "./data/cache.db")
        self.encrypt: bool = bool(raw.get("encrypt", True))


class LoggingConfig:
    def __init__(self, raw: dict) -> None:
        self.level: str = raw.get("level", "INFO").upper()
        self.format: str = raw.get("format", "json")


class MissionsConfig:
    def __init__(self, raw: dict) -> None:
        self.enabled: bool = bool(raw.get("enabled", True))
        self.skills_path: str = raw.get("skills_path", "./skills.yaml")


class VisionConfig:
    def __init__(self, raw: dict) -> None:
        self.enabled: bool = bool(raw.get("enabled", False))
        self.base_url: str = str(raw.get("base_url", "https://api.openai.com/v1")).rstrip("/")
        self.model: str = str(raw.get("model", "gpt-4o-mini"))
        self.api_key_env: str = str(raw.get("api_key_env", "OPENAI_API_KEY"))
        self.timeout_seconds: int = int(raw.get("timeout_seconds", 60))
        self.upload_dir: str = str(raw.get("upload_dir", "./data/vision"))
        self.max_image_bytes: int = int(raw.get("max_image_bytes", 10 * 1024 * 1024))

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


class AppConfig:
    def __init__(self, raw: dict) -> None:
        self.server = ServerConfig(raw.get("server", {}))
        self.provider = ProviderConfig(raw.get("provider", {}))
        self.cache = CacheConfig(raw.get("cache", {}))
        self.logging = LoggingConfig(raw.get("logging", {}))
        self.missions = MissionsConfig(raw.get("missions", {}))
        self.vision = VisionConfig(raw.get("vision", {}))

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            return cls({})
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls(raw)

    @classmethod
    def from_env(cls) -> "AppConfig":
        _load_dotenv_files()
        config_path = os.environ.get("ASLOR_CONFIG", "config.yaml")
        return cls.from_file(config_path)


def _load_dotenv_files() -> None:
    """Load environment variables from local dotenv files if present."""
    candidates = (
        ".env",
        ".env.local",
    )
    for name in candidates:
        path = Path(name)
        if path.exists():
            load_dotenv(path, override=True)
