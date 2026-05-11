"""Skill registry -- loaded from ``skills.yaml`` at startup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from aslor.missions.models import SkillDefinition


class SkillRegistry:
    """In-memory registry of skill definitions loaded from skills.yaml."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def load(self, path: str | Path) -> None:
        raw = _read_yaml(path)
        skills_raw = raw.get("skills", {})
        for sid, data in skills_raw.items():
            self._skills[sid] = SkillDefinition(
                id=sid,
                title=data.get("title", sid),
                description=data.get("description", ""),
                system_prompt=data.get("system_prompt", ""),
                enabled=bool(data.get("enabled", True)),
            )

    def get_skill(self, skill_id: str) -> SkillDefinition | None:
        return self._skills.get(skill_id)

    def list_skills(self, enabled_only: bool = False) -> list[SkillDefinition]:
        if enabled_only:
            return [s for s in self._skills.values() if s.enabled]
        return list(self._skills.values())

    def enable_skill(self, skill_id: str, enabled: bool) -> None:
        s = self._skills.get(skill_id)
        if s is None:
            return
        self._skills[skill_id] = SkillDefinition(
            id=s.id,
            title=s.title,
            description=s.description,
            system_prompt=s.system_prompt,
            enabled=enabled,
        )


_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _registry
    if _registry is None:
        _registry = SkillRegistry()
        _registry.load(_default_path())
    return _registry


def _default_path() -> Path:
    env = os.environ.get("ASLOR_SKILLS", "skills.yaml")
    return Path(env)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}
