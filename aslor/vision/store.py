"""Vision asset and analysis storage."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from typing import Any

from aslor.cache.db import CacheDB


_IMAGE_PREFIX = "vision:image:"
_ANALYSIS_PREFIX = "vision:analysis:"


class VisionStore:
    def __init__(self, db: CacheDB, upload_dir: str | Path) -> None:
        self._db = db
        self._upload_dir = Path(upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        content: bytes,
        filename: str = "",
        mime_type: str = "",
    ) -> dict[str, Any]:
        sha256 = hashlib.sha256(content).hexdigest()
        image_id = sha256[:24]
        ext = self._extension_for(filename, mime_type)
        disk_path = self._upload_dir / f"{image_id}{ext}"
        if not disk_path.exists():
            disk_path.write_bytes(content)

        detected_mime = mime_type or mimetypes.guess_type(str(disk_path))[0] or "application/octet-stream"
        meta = {
            "image_id": image_id,
            "sha256": sha256,
            "filename": filename or disk_path.name,
            "mime_type": detected_mime,
            "path": str(disk_path),
            "size_bytes": len(content),
        }
        self._db.set(f"{_IMAGE_PREFIX}{image_id}", meta)
        return meta

    def load_image(self, image_id: str) -> dict[str, Any] | None:
        raw = self._db.get(f"{_IMAGE_PREFIX}{image_id}")
        return raw if isinstance(raw, dict) else None

    def load_image_bytes(self, image_id: str) -> bytes | None:
        meta = self.load_image(image_id)
        if not meta:
            return None
        path = meta.get("path")
        if not isinstance(path, str):
            return None
        file_path = Path(path)
        if not file_path.exists():
            return None
        return file_path.read_bytes()

    def get_data_url(self, image_id: str) -> str | None:
        meta = self.load_image(image_id)
        data = self.load_image_bytes(image_id)
        if not meta or data is None:
            return None
        mime = str(meta.get("mime_type") or "application/octet-stream")
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def load_analysis(self, image_hash: str) -> dict[str, Any] | None:
        raw = self._db.get(f"{_ANALYSIS_PREFIX}{image_hash}")
        return raw if isinstance(raw, dict) else None

    def save_analysis(self, image_hash: str, analysis: dict[str, Any]) -> None:
        self._db.set(f"{_ANALYSIS_PREFIX}{image_hash}", analysis)

    @staticmethod
    def _extension_for(filename: str, mime_type: str) -> str:
        suffix = Path(filename).suffix
        if suffix:
            return suffix
        guessed = mimetypes.guess_extension(mime_type or "")
        return guessed or ".bin"
