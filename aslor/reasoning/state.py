"""ReasoningStateStore - CRUD layer on top of CacheDB.

Keys are session identifiers derived from the conversation fingerprint.
Values are either legacy plain reasoning-state dicts or a richer payload that
tracks the latest state plus per-assistant-message history for longer chats.
"""

from __future__ import annotations

import copy
from typing import Any

from aslor.cache.db import CacheDB


_KEY_PREFIX = "reasoning:"


class ReasoningStateStore:
    def __init__(self, db: CacheDB) -> None:
        self._db = db

    def _key(self, session_id: str) -> str:
        return f"{_KEY_PREFIX}{session_id}"

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist reasoning state for a session."""
        self._db.set(self._key(session_id), state)

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve the latest reasoning state for a session, or None if absent."""
        raw = self._db.get(self._key(session_id))
        if raw is None:
            return None
        if isinstance(raw, dict) and "latest" in raw and "history" in raw:
            latest = raw.get("latest")
            return copy.deepcopy(latest) if isinstance(latest, dict) else None
        return raw if isinstance(raw, dict) else None

    def load_history(self, session_id: str) -> list[dict[str, Any]]:
        """Return stored per-assistant reasoning history for a session."""
        raw = self._db.get(self._key(session_id))
        if not isinstance(raw, dict):
            return []
        history = raw.get("history")
        if not isinstance(history, list):
            return []
        return [copy.deepcopy(item) for item in history if isinstance(item, dict)]

    def find_message_states(
        self,
        assistant_keys: list[str],
        exclude_session_id: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Find exact assistant-message states across all cached sessions.

        The most recently updated matching session wins for each assistant key.
        """
        wanted = {key for key in assistant_keys if key}
        if not wanted:
            return {}

        matches: dict[str, dict[str, Any]] = {}
        for entry in self._db.list_entries(prefix=_KEY_PREFIX):
            session_id = entry["key"][len(_KEY_PREFIX):]
            if exclude_session_id and session_id == exclude_session_id:
                continue
            raw = self._db.get(entry["key"])
            if not isinstance(raw, dict):
                continue
            history = raw.get("history")
            if not isinstance(history, list):
                continue
            for item in history:
                if not isinstance(item, dict):
                    continue
                assistant_key = item.get("assistant_key")
                state = item.get("state")
                if (
                    isinstance(assistant_key, str)
                    and assistant_key in wanted
                    and assistant_key not in matches
                    and isinstance(state, dict)
                ):
                    matches[assistant_key] = copy.deepcopy(state)
            if wanted.issubset(matches.keys()):
                break
        return matches

    def append_message_state(
        self,
        session_id: str,
        assistant_key: str,
        state: dict[str, Any],
    ) -> None:
        """Persist the latest state and associate it with one assistant message."""
        raw = self._db.get(self._key(session_id))
        history: list[dict[str, Any]] = []

        if isinstance(raw, dict) and "history" in raw and "latest" in raw:
            existing = raw.get("history")
            if isinstance(existing, list):
                history = [copy.deepcopy(item) for item in existing if isinstance(item, dict)]

        updated = False
        for item in history:
            if item.get("assistant_key") == assistant_key:
                item["state"] = copy.deepcopy(state)
                updated = True
                break

        if not updated:
            history.append(
                {
                    "assistant_key": assistant_key,
                    "state": copy.deepcopy(state),
                }
            )

        self._db.set(
            self._key(session_id),
            {
                "latest": copy.deepcopy(state),
                "history": history,
            },
        )

    def delete(self, session_id: str) -> None:
        self._db.delete(self._key(session_id))

    def list_entries(self) -> list[dict[str, Any]]:
        """List all reasoning cache entries with metadata."""
        return self._db.list_entries(prefix=_KEY_PREFIX)

    def clear_all(self) -> int:
        return self._db.clear()

    def count(self) -> int:
        return self._db.count()
