"""Vision analysis sidecar for multimodal screenshots."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import httpx

from aslor.config import VisionConfig
from aslor.vision.store import VisionStore

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You analyze Android app screenshots and UI mockups. "
    "Return strict JSON only with keys: screen_type, summary, visible_text, "
    "layout_issues, accessibility_issues, component_tree. "
    "Each issue item must include severity and issue. "
    "Keep content concise and grounded in what is visible."
)


class VisionAnalyzer:
    def __init__(self, config: VisionConfig, store: VisionStore) -> None:
        self._config = config
        self._store = store

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    async def analyze(self, image_urls: list[str], prompt: str) -> dict[str, Any] | None:
        if not self.enabled or not image_urls:
            return None

        cache_key = self._analysis_key(image_urls, prompt)
        cached = self._store.load_analysis(cache_key)
        if cached is not None:
            return cached

        api_key = self._config.api_key
        if not api_key:
            logger.warning("vision: enabled but no API key available in %s", self._config.api_key_env)
            return None

        payload = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_content(prompt, image_urls),
                },
            ],
            "stream": False,
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._config.base_url}/chat/completions"
        try:
            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("vision: analysis request failed - %s", exc)
            return None

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except Exception:
            logger.warning("vision: malformed analysis response")
            return None

        analysis = _parse_json_content(content)
        if analysis is None:
            logger.warning("vision: response was not valid JSON")
            return None

        self._store.save_analysis(cache_key, analysis)
        return analysis

    def resolve_message_image_urls(self, messages: list[dict[str, Any]]) -> list[str]:
        urls: list[str] = []
        seen: set[str] = set()
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict) or part.get("type") != "image_url":
                        continue
                    value = self._extract_image_url(part.get("image_url"))
                    resolved = self._resolve_url(value)
                    if resolved and resolved not in seen:
                        seen.add(resolved)
                        urls.append(resolved)
            elif isinstance(content, str):
                for ref in _extract_inline_image_refs(content):
                    resolved = self._resolve_url(ref)
                    if resolved and resolved not in seen:
                        seen.add(resolved)
                        urls.append(resolved)
        return urls

    @staticmethod
    def build_visual_context(analysis: dict[str, Any]) -> str:
        lines: list[str] = ["VISUAL_CONTEXT:"]
        summary = str(analysis.get("summary") or "").strip()
        screen_type = str(analysis.get("screen_type") or "").strip()
        if screen_type:
            lines.append(f"- Screen: {screen_type}")
        if summary:
            lines.append(f"- Summary: {summary}")

        visible_text = analysis.get("visible_text")
        if isinstance(visible_text, list) and visible_text:
            preview = ", ".join(str(item) for item in visible_text[:8])
            lines.append(f"- Visible text: {preview}")

        layout_issues = _format_issues("Layout issues", analysis.get("layout_issues"))
        if layout_issues:
            lines.extend(layout_issues)

        accessibility_issues = _format_issues("Accessibility issues", analysis.get("accessibility_issues"))
        if accessibility_issues:
            lines.extend(accessibility_issues)

        return "\n".join(lines)

    def _resolve_url(self, value: str | None) -> str | None:
        if not value:
            return None
        if value.startswith("aslor://image/"):
            image_id = value.removeprefix("aslor://image/")
            return self._store.get_data_url(image_id)
        if value.startswith("data:") or value.startswith("http://") or value.startswith("https://"):
            return value
        return None

    @staticmethod
    def _extract_image_url(image_url_part: Any) -> str | None:
        if isinstance(image_url_part, str):
            return image_url_part
        if isinstance(image_url_part, dict):
            url = image_url_part.get("url")
            return str(url) if url else None
        return None

    @staticmethod
    def _build_content(prompt: str, image_urls: list[str]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        return content

    @staticmethod
    def _analysis_key(image_urls: list[str], prompt: str) -> str:
        raw = json.dumps({"images": image_urls, "prompt": prompt}, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_inline_image_refs(text: str) -> list[str]:
    refs: list[str] = []
    marker = "aslor://image/"
    start = 0
    while True:
        idx = text.find(marker, start)
        if idx == -1:
            return refs
        end = idx + len(marker)
        while end < len(text) and text[end] not in " \n\r\t)]}\"'":
            end += 1
        refs.append(text[idx:end])
        start = end


def _parse_json_content(content: Any) -> dict[str, Any] | None:
    if isinstance(content, list):
        text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    else:
        text = str(content or "")
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_issues(title: str, raw: Any) -> list[str]:
    if not isinstance(raw, list) or not raw:
        return []
    lines = [f"- {title}:"]
    for item in raw[:5]:
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or "info")
        issue = str(item.get("issue") or "").strip()
        if issue:
            lines.append(f"  {severity}: {issue}")
    return lines
