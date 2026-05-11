from __future__ import annotations

from pathlib import Path

from aslor.config import AppConfig


def test_from_env_loads_dotenv(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOTENV_TEST_KEY", raising=False)
    monkeypatch.delenv("ASLOR_CONFIG", raising=False)

    (tmp_path / ".env").write_text("DOTENV_TEST_KEY=loaded-from-dotenv\n", encoding="utf-8")
    (tmp_path / "config.yaml").write_text(
        "provider:\n  api_key_env: DOTENV_TEST_KEY\n",
        encoding="utf-8",
    )

    cfg = AppConfig.from_env()

    assert cfg.provider.api_key == "loaded-from-dotenv"
