"""Tests for CacheDB — encrypted and plain SQLite storage."""

from __future__ import annotations

import pytest
from aslor.cache.db import CacheDB


class TestPlainCache:
    def test_set_and_get(self, tmp_db: CacheDB) -> None:
        tmp_db.set("key1", {"foo": "bar"})
        result = tmp_db.get("key1")
        assert result == {"foo": "bar"}

    def test_overwrite(self, tmp_db: CacheDB) -> None:
        tmp_db.set("key1", {"v": 1})
        tmp_db.set("key1", {"v": 2})
        assert tmp_db.get("key1") == {"v": 2}

    def test_missing_key_returns_none(self, tmp_db: CacheDB) -> None:
        assert tmp_db.get("nonexistent") is None

    def test_delete(self, tmp_db: CacheDB) -> None:
        tmp_db.set("k", {"x": 1})
        tmp_db.delete("k")
        assert tmp_db.get("k") is None

    def test_clear_returns_count(self, tmp_db: CacheDB) -> None:
        tmp_db.set("a", 1)
        tmp_db.set("b", 2)
        deleted = tmp_db.clear()
        assert deleted == 2
        assert tmp_db.count() == 0

    def test_count(self, tmp_db: CacheDB) -> None:
        assert tmp_db.count() == 0
        tmp_db.set("x", "y")
        assert tmp_db.count() == 1

    def test_stores_various_types(self, tmp_db: CacheDB) -> None:
        tmp_db.set("str", "hello")
        tmp_db.set("int", 42)
        tmp_db.set("list", [1, 2, 3])
        assert tmp_db.get("str") == "hello"
        assert tmp_db.get("int") == 42
        assert tmp_db.get("list") == [1, 2, 3]


class TestEncryptedCache:
    def test_set_and_get_encrypted(self, tmp_db_encrypted: CacheDB) -> None:
        tmp_db_encrypted.set("sec", {"token": "abc123"})
        assert tmp_db_encrypted.get("sec") == {"token": "abc123"}

    def test_encrypted_value_not_plaintext(self, tmp_db_encrypted: CacheDB, tmp_path) -> None:
        import sqlite3

        tmp_db_encrypted.set("secret", {"reasoning_content": "my chain of thought"})
        raw_conn = sqlite3.connect(str(tmp_path / "test_enc.db"))
        row = raw_conn.execute("SELECT value FROM kv WHERE key='secret'").fetchone()
        raw_conn.close()
        assert row is not None
        raw_bytes: bytes = row[0]
        assert b"my chain of thought" not in raw_bytes
