"""Encrypted SQLite wrapper.

All values are optionally encrypted with Fernet symmetric encryption before
being written, and decrypted on read.  The encryption key is derived from a
machine-stable secret stored next to the database file (never in the DB
itself).  If encryption is disabled, values are stored as plain JSON strings.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any


def _get_or_create_key(key_path: Path) -> bytes:
    """Return Fernet key bytes, creating and persisting if absent."""
    from cryptography.fernet import Fernet

    if key_path.exists():
        return key_path.read_bytes().strip()
    key = Fernet.generate_key()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_bytes(key)
    return key


class CacheDB:
    """Thread-safe SQLite key/value store with optional Fernet encryption."""

    def __init__(self, db_path: str | Path, encrypt: bool = True) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._encrypt = encrypt
        self._fernet = None
        if encrypt:
            from cryptography.fernet import Fernet

            key_path = self._path.with_suffix(".key")
            key = _get_or_create_key(key_path)
            self._fernet = Fernet(key)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self._path), check_same_thread=False)
        return self._local.conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kv (
                    key   TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    updated_at INTEGER DEFAULT (strftime('%s','now'))
                )
                """
            )
            conn.commit()

    def set(self, key: str, value: Any) -> None:
        serialized = json.dumps(value).encode()
        if self._fernet:
            stored = self._fernet.encrypt(serialized)
        else:
            stored = serialized
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv (key, value, updated_at) VALUES (?, ?, strftime('%s','now'))",
                (key, stored),
            )
            conn.commit()

    def get(self, key: str) -> Any | None:
        row = self._conn().execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        raw: bytes = row[0]
        if self._fernet:
            try:
                raw = self._fernet.decrypt(raw)
            except Exception:
                return None
        return json.loads(raw)

    def delete(self, key: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM kv WHERE key = ?", (key,))
            conn.commit()

    def clear(self) -> int:
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM kv")
            conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        row = self._conn().execute("SELECT COUNT(*) FROM kv").fetchone()
        return row[0] if row else 0

    def list_entries(self, prefix: str = "") -> list[dict[str, Any]]:
        """Return all cache entries with key, size, and last-updated timestamp."""
        rows = self._conn().execute(
            "SELECT key, value, updated_at FROM kv ORDER BY updated_at DESC"
        ).fetchall()
        entries: list[dict[str, Any]] = []
        for key, raw, ts in rows:
            if prefix and not key.startswith(prefix):
                continue
            size = len(raw)
            entries.append({"key": key, "size_bytes": size, "updated_at": int(ts or 0)})
        return entries

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
