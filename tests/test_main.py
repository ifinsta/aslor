from __future__ import annotations

from aslor.main import _env_flag, _reload_options


def test_reload_options_enabled_contains_env_and_yaml():
    opts = _reload_options(True)

    assert opts["reload"] is True
    assert ".env" in opts["reload_includes"]
    assert ".env.*" in opts["reload_includes"]
    assert "*.yaml" in opts["reload_includes"]
    assert "*.py" in opts["reload_includes"]


def test_reload_options_disabled_is_empty():
    assert _reload_options(False) == {}


def test_env_flag_truthy(monkeypatch):
    monkeypatch.setenv("ASLOR_RELOAD", "true")
    assert _env_flag("ASLOR_RELOAD") is True


def test_env_flag_falsey(monkeypatch):
    monkeypatch.setenv("ASLOR_RELOAD", "0")
    assert _env_flag("ASLOR_RELOAD") is False
