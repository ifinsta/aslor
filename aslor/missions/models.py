"""Mission and skill data models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class SkillDefinition:
    """Loaded from skills.yaml -- toggleable at runtime via the dashboard."""

    id: str
    title: str
    description: str
    system_prompt: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "enabled": self.enabled,
        }


@dataclass
class MissionState:
    """Runtime state of a single mission, persisted via CacheDB.

    A mission represents a project goal the proxy drives toward.
    Only one mission is active at a time; creating a new active
    mission auto-pauses any previously active one.
    """

    mission_id: str
    title: str
    description: str
    success_criteria: str
    max_attempts: int
    status: str  # "active" | "completed" | "failed" | "paused"
    current_attempt: int
    created_at: int
    updated_at: int
    completed_at: int | None = None
    notes: str = ""
    skill_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "title": self.title,
            "description": self.description,
            "success_criteria": self.success_criteria,
            "max_attempts": self.max_attempts,
            "status": self.status,
            "current_attempt": self.current_attempt,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "notes": self.notes,
            "skill_ids": self.skill_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionState:
        return cls(
            mission_id=data["mission_id"],
            title=data.get("title", ""),
            description=data.get("description", ""),
            success_criteria=data.get("success_criteria", ""),
            max_attempts=int(data.get("max_attempts", 3)),
            status=data.get("status", "active"),
            current_attempt=int(data.get("current_attempt", 0)),
            created_at=int(data.get("created_at", 0)),
            updated_at=int(data.get("updated_at", 0)),
            completed_at=data.get("completed_at"),
            notes=data.get("notes", ""),
            skill_ids=data.get("skill_ids"),
        )
