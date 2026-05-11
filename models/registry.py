"""Model & provider registry — loaded from ``models.yaml`` at startup.

This is the single source of truth for every model and provider the proxy
knows about.  Adding a model is a one-line YAML change; no code edits needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelEntry:
    id: str
    reasoning: bool = False
    reasoning_field: str = ""
    streaming: bool = True

    def to_model_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "model",
            "created": 1700000000,
            "owned_by": "",
        }


@dataclass
class ProviderEntry:
    name: str
    base_url: str = ""
    api_key_env: str = ""
    timeout_seconds: int = 120
    adapter: str = "passthrough"
    thinking_enabled: bool = False
    thinking_budget_tokens: int = 5000
    models: list[ModelEntry] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.models is None:
            self.models = []

    def to_presets(self) -> dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "thinking_enabled": self.thinking_enabled,
            "thinking_budget_tokens": self.thinking_budget_tokens,
        }


class ModelRegistry:
    """In-memory registry of all providers and their models."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderEntry] = {}
        self._models: dict[str, ModelEntry] = {}
        self._prefixes: list[tuple[str, ModelEntry]] = []

    # ── loading ────────────────────────────────────────────────────────────

    def load(self, path: str | Path) -> None:
        raw = _read_yaml(path)
        providers_raw = raw.get("providers", {})
        for name, data in providers_raw.items():
            self._load_provider(name, data)

    def _load_provider(self, name: str, data: dict[str, Any]) -> None:
        model_entries: list[ModelEntry] = []
        for m in data.get("models", []):
            entry = ModelEntry(
                id=m["id"],
                reasoning=bool(m.get("reasoning", False)),
                reasoning_field=m.get("reasoning_field", ""),
                streaming=bool(m.get("streaming", True)),
            )
            model_entries.append(entry)
            self._models[entry.id] = entry
            self._prefixes.append((entry.id.lower(), entry))

        provider = ProviderEntry(
            name=name,
            base_url=str(data.get("base_url", "")).rstrip("/"),
            api_key_env=str(data.get("api_key_env", "")),
            timeout_seconds=int(data.get("timeout_seconds", 120)),
            adapter=str(data.get("adapter", "passthrough")),
            thinking_enabled=bool(data.get("thinking_enabled", False)),
            thinking_budget_tokens=int(data.get("thinking_budget_tokens", 5000)),
            models=model_entries,
        )
        self._providers[name] = provider
        # Keep prefix list sorted longest-first for precise matching.
        self._prefixes.sort(key=lambda x: -len(x[0]))

    # ── lookup ─────────────────────────────────────────────────────────────

    def get_model(self, model_name: str) -> ModelEntry | None:
        """Exact-match lookup."""
        return self._models.get(model_name)

    def find_model(self, model_name: str) -> ModelEntry | None:
        """Prefix-match lookup (longest prefix wins)."""
        lower = model_name.lower()
        for prefix, entry in self._prefixes:
            if lower.startswith(prefix):
                return entry
        return None

    def get_provider(self, name: str) -> ProviderEntry | None:
        return self._providers.get(name)

    def list_models(self, provider_name: str | None = None) -> list[ModelEntry]:
        """Return all models, optionally filtered by provider name."""
        if provider_name:
            p = self._providers.get(provider_name)
            return list(p.models) if p else []
        return list(self._models.values())

    def list_providers(self) -> list[ProviderEntry]:
        return list(self._providers.values())

    def get_provider_presets(self, name: str) -> dict[str, Any] | None:
        p = self._providers.get(name)
        return p.to_presets() if p else None

    def get_all_presets(self) -> dict[str, dict[str, Any]]:
        return {name: p.to_presets() for name, p in self._providers.items()}


# ── Singleton ────────────────────────────────────────────────────────────

_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
        _registry.load(_default_path())
    return _registry


def _default_path() -> Path:
    env = os.environ.get("ASLOR_MODELS", "models.yaml")
    return Path(env)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ── Backwards-compatible function API ─────────────────────────────────────

def get_capability(model_name: str) -> ModelEntry:
    """Return the :class:`ModelEntry` for *model_name*, or a default empty entry."""
    reg = get_registry()
    entry = reg.find_model(model_name)
    if entry is not None:
        return entry
    return ModelEntry(id=model_name)
