"""Mission state persistence via CacheDB."""

from __future__ import annotations

import time
from typing import Any

from aslor.cache.db import CacheDB
from aslor.missions.models import MissionState

_KEY_PREFIX = "mission:"


class MissionStateStore:
    """CRUD for mission state stored in CacheDB with ``mission:`` prefix."""

    def __init__(self, db: CacheDB) -> None:
        self._db = db

    def _key(self, mission_id: str) -> str:
        return f"{_KEY_PREFIX}{mission_id}"

    def create(self, mission: MissionState) -> None:
        self._db.set(self._key(mission.mission_id), mission.to_dict())

    def save(self, mission: MissionState) -> None:
        mission.updated_at = int(time.time())
        self._db.set(self._key(mission.mission_id), mission.to_dict())

    def load(self, mission_id: str) -> MissionState | None:
        raw = self._db.get(self._key(mission_id))
        if raw is None:
            return None
        return MissionState.from_dict(raw)

    def delete(self, mission_id: str) -> None:
        self._db.delete(self._key(mission_id))

    def list_all(self) -> list[MissionState]:
        entries = self._db.list_entries(prefix=_KEY_PREFIX)
        missions: list[MissionState] = []
        for entry in entries:
            raw = self._db.get(entry["key"])
            if raw is not None:
                missions.append(MissionState.from_dict(raw))
        return missions

    def get_active_mission(self) -> MissionState | None:
        missions = self.list_all()
        for m in missions:
            if m.status == "active":
                return m
        return None

    def reset(self, mission_id: str) -> MissionState | None:
        mission = self.load(mission_id)
        if mission is None:
            return None
        mission.current_attempt = 0
        mission.status = "active"
        mission.completed_at = None
        mission.notes = ""
        self.save(mission)
        return mission

    def count(self) -> int:
        all_entries = self._db.list_entries(prefix=_KEY_PREFIX)
        return len(all_entries)
