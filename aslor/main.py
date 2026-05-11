"""Entry point - run with ``python -m aslor.main``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import uvicorn

from aslor.config import AppConfig


def main() -> None:
    args = _parse_args()
    config = AppConfig.from_env()
    reload_enabled = args.reload or _env_flag("ASLOR_RELOAD")

    uvicorn.run(
        "aslor.server.app:create_app" if reload_enabled else _load_app(),
        host=config.server.host,
        port=config.server.port,
        log_config=None,
        factory=reload_enabled,
        **_reload_options(reload_enabled),
    )


def _load_app():
    from aslor.server.app import create_app

    return create_app(AppConfig.from_env())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ASLOR proxy server.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on Python, config, model, skill, and .env file changes.",
    )
    return parser.parse_args()


def _reload_options(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {}

    watch_roots = [str(Path.cwd()), str(Path(__file__).resolve().parent)]
    return {
        "reload": True,
        "reload_dirs": watch_roots,
        "reload_includes": [
            "*.py",
            "*.yaml",
            "*.yml",
            ".env",
            ".env.*",
        ],
    }


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    main()
