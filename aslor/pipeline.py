"""RequestPipelineAgent — orchestrates the full request lifecycle.

Wires all sub-agents together without containing business logic itself.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from typing import Any, AsyncGenerator

import httpx
from fastapi import Request
from fastapi.responses import StreamingResponse

import time as _time

from aslor.agents.capture import capture_from_assembled
from aslor.agents.forwarder import forward
from aslor.agents.redaction import redact_for_log
from aslor.agents.relay import relay_stream
from aslor.config import AppConfig
from aslor.missions.evaluator import EvaluationResult, evaluate_response
from aslor.missions.injector import inject_mission_context
from aslor.missions.models import SkillDefinition
from aslor.missions.registry import SkillRegistry
from aslor.missions.state import MissionStateStore
from aslor.providers.registry import get_adapter
from aslor.reasoning.detector import detect
from aslor.reasoning.repair import repair_messages
from aslor.reasoning.state import ReasoningStateStore
from aslor.server.stats import StatsTracker
from aslor.vision.analyzer import VisionAnalyzer

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised by any stage to signal a recoverable pipeline failure."""


class RequestPipeline:
    def __init__(
        self,
        config: AppConfig,
        store: ReasoningStateStore,
        stats: StatsTracker | None = None,
        mission_store: MissionStateStore | None = None,
        skill_registry: SkillRegistry | None = None,
        vision_analyzer: VisionAnalyzer | None = None,
        upstream_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._stats = stats
        self._mission_store = mission_store
        self._skill_registry = skill_registry
        self._vision_analyzer = vision_analyzer
        self._upstream_client = upstream_client

    async def run(
        self,
        body: dict[str, Any],
        api_key: str | None = None,
        *,
        session_hint: str | None = None,
    ) -> StreamingResponse:
        """Execute the full pipeline and return a FastAPI response."""
        t0 = _time.time()
        requested_stream = bool(body.get("stream", False))
        model = str(body.get("model", "")).strip()
        if not model:
            model = self._config.provider.default_model.strip()
            if model:
                body["model"] = model

        logger.debug("pipeline: incoming request model=%s", body.get("model"))
        logger.debug("pipeline: body (redacted) = %s", redact_for_log(body))

        detection = detect(body, self._config, session_hint=session_hint)
        adapter = get_adapter(self._config, detection.model_name)
        has_key = bool(api_key)

        logger.info(
            "pipeline: model=%s needs_repair=%s session=%s messages=%d",
            model, detection.needs_repair, detection.session_key[:8],
            len(body.get("messages", [])),
        )

        messages_raw = body.get("messages", [])
        if not isinstance(messages_raw, list):
            messages_raw = []
        messages = [dict(m) for m in messages_raw if isinstance(m, dict)]

        repair_applied = False
        if detection.needs_repair:
            messages = repair_messages(
                messages=messages,
                session_id=detection.session_key,
                adapter=adapter,
                store=self._store,
            )
            repair_applied = True

        body["messages"] = messages

        if self._vision_analyzer and self._vision_analyzer.enabled:
            messages = await self._inject_visual_context(messages)
            body["messages"] = messages

        # Stage: inject active mission context as a system message
        if self._mission_store and self._skill_registry:
            active_mission = self._mission_store.get_active_mission()
            if active_mission:
                bound_skills: list[SkillDefinition] = []
                if active_mission.skill_ids:
                    for sid in active_mission.skill_ids:
                        s = self._skill_registry.get_skill(sid)
                        if s and s.enabled:
                            bound_skills.append(s)
                messages = inject_mission_context(messages, active_mission, bound_skills)
                body["messages"] = messages

        normalized = adapter.normalize_request(body)

        try:
            response = await forward(
                normalized,
                adapter,
                self._config,
                api_key=api_key,
                client=self._upstream_client,
            )
        except httpx.HTTPError as exc:
            logger.error("pipeline: upstream transport error — %s", exc)
            if self._stats:
                self._stats.record(
                    method="POST", path="/v1/chat/completions",
                    status_code=502,
                    latency_ms=round((_time.time() - t0) * 1000),
                    model=model, repair_applied=repair_applied,
                    api_key_available=has_key,
                )
            return StreamingResponse(
                content=_single_chunk(
                    json.dumps({"error": {"message": str(exc), "type": "upstream_unreachable"}}).encode()
                ),
                status_code=502,
                media_type="application/json",
            )

        if response.status_code >= 400:
            error_body = await response.aread()
            await response.aclose()
            retried_ok = False
            retried_context_ok = False
            upstream_status = response.status_code

            if response.status_code == 400 and _is_context_length_error(error_body):
                max_tokens, _requested = _parse_context_length_limits(error_body)
                budget_chars = int((max_tokens or 1048576) * 4 * 0.9)
                ctx_attempts: list[list[dict[str, Any]]] = []
                ctx_attempts.append(_strip_reasoning_fields(messages))
                ctx_attempts.append(_trim_messages_to_char_budget(messages, budget_chars))

                for attempt_messages in ctx_attempts:
                    retry_body = copy.deepcopy(body)
                    retry_body["messages"] = attempt_messages
                    retry_normalized = adapter.normalize_request(retry_body)
                    logger.warning(
                        "pipeline: upstream context overflow; retrying with reduced history (session=%s, approx_chars=%d)",
                        detection.session_key[:8],
                        _estimate_message_chars(attempt_messages),
                    )
                    try:
                        retry_response = await forward(
                            retry_normalized,
                            adapter,
                            self._config,
                            api_key=api_key,
                            client=self._upstream_client,
                        )
                    except httpx.HTTPError as exc:
                        logger.error("pipeline: upstream transport error (context retry) — %s", exc)
                        break

                    if retry_response.status_code < 400:
                        response = retry_response
                        normalized = retry_normalized
                        retried_context_ok = True
                        break

                    retry_error = await retry_response.aread()
                    await retry_response.aclose()
                    if retry_response.status_code == 400 and _is_context_length_error(retry_error):
                        error_body = retry_error
                        continue

                    error_body = retry_error
                    if self._stats:
                        self._stats.record(
                            method="POST", path="/v1/chat/completions",
                            status_code=retry_response.status_code,
                            latency_ms=round((_time.time() - t0) * 1000),
                            model=model, repair_applied=True,
                            api_key_available=has_key,
                        )
                    return StreamingResponse(
                        content=_single_chunk(error_body),
                        status_code=retry_response.status_code,
                        media_type="application/json",
                    )
            if (
                response.status_code == 400
                and _is_missing_reasoning_error(error_body)
                and adapter.reasoning_field()
                and any(m.get("role") == "assistant" for m in messages)
                and not retried_context_ok
            ):
                retry_messages = _strip_assistant_history(messages)
                retry_body = copy.deepcopy(body)
                retry_body["messages"] = retry_messages
                retry_normalized = adapter.normalize_request(retry_body)
                logger.warning(
                    "pipeline: upstream rejected missing %s; retrying with stripped assistant history (session=%s)",
                    adapter.reasoning_field(),
                    detection.session_key[:8],
                )
                try:
                    retry_response = await forward(
                        retry_normalized,
                        adapter,
                        self._config,
                        api_key=api_key,
                        client=self._upstream_client,
                    )
                except httpx.HTTPError as exc:
                    logger.error("pipeline: upstream transport error (retry) — %s", exc)
                    return StreamingResponse(
                        content=_single_chunk(
                            json.dumps({"error": {"message": str(exc), "type": "upstream_unreachable"}}).encode()
                        ),
                        status_code=502,
                        media_type="application/json",
                    )

                if retry_response.status_code < 400:
                    response = retry_response
                    normalized = retry_normalized
                    retried_ok = True
                else:
                    retry_error = await retry_response.aread()
                    await retry_response.aclose()
                    if retry_response.status_code == 400 and _is_missing_reasoning_error(retry_error):
                        retry2_body = copy.deepcopy(retry_body)
                        _strip_reasoning_mode_params(retry2_body)
                        retry2_normalized = adapter.normalize_request(retry2_body)
                        logger.warning(
                            "pipeline: missing %s persisted; retrying with thinking mode disabled (session=%s)",
                            adapter.reasoning_field(),
                            detection.session_key[:8],
                        )
                        try:
                            retry2_response = await forward(
                                retry2_normalized,
                                adapter,
                                self._config,
                                api_key=api_key,
                                client=self._upstream_client,
                            )
                        except httpx.HTTPError as exc:
                            logger.error("pipeline: upstream transport error (retry2) — %s", exc)
                            return StreamingResponse(
                                content=_single_chunk(
                                    json.dumps({"error": {"message": str(exc), "type": "upstream_unreachable"}}).encode()
                                ),
                                status_code=502,
                                media_type="application/json",
                            )

                        if retry2_response.status_code < 400:
                            response = retry2_response
                            normalized = retry2_normalized
                            retried_ok = True
                        else:
                            retry2_error = await retry2_response.aread()
                            await retry2_response.aclose()
                            logger.error(
                                "pipeline: upstream returned %d on retry2 — %s",
                                retry2_response.status_code,
                                retry2_error.decode()[:500],
                            )
                            error_body = retry2_error
                            if self._stats:
                                self._stats.record(
                                    method="POST", path="/v1/chat/completions",
                                    status_code=retry2_response.status_code,
                                    latency_ms=round((_time.time() - t0) * 1000),
                                    model=model, repair_applied=True,
                                    api_key_available=has_key,
                                )
                            return StreamingResponse(
                                content=_single_chunk(error_body),
                                status_code=retry2_response.status_code,
                                media_type="application/json",
                            )
                    if not retried_ok:
                        logger.error(
                            "pipeline: upstream returned %d on retry — %s",
                            retry_response.status_code,
                            retry_error.decode()[:500],
                        )
                        error_body = retry_error
                        if self._stats:
                            self._stats.record(
                                method="POST", path="/v1/chat/completions",
                                status_code=retry_response.status_code,
                                latency_ms=round((_time.time() - t0) * 1000),
                                model=model, repair_applied=True,
                                api_key_available=has_key,
                            )
                        return StreamingResponse(
                            content=_single_chunk(error_body),
                            status_code=retry_response.status_code,
                            media_type="application/json",
                        )

            if not (retried_ok or retried_context_ok):
                unknown_param = _extract_unknown_parameter(error_body)
                if unknown_param:
                    retry_body = copy.deepcopy(body)
                    _delete_path_key(retry_body, unknown_param)
                    retry_normalized = adapter.normalize_request(retry_body)
                    logger.warning(
                        "pipeline: upstream rejected unknown parameter '%s'; retrying without it (session=%s)",
                        unknown_param,
                        detection.session_key[:8],
                    )
                    try:
                        retry_response = await forward(
                            retry_normalized,
                            adapter,
                            self._config,
                            api_key=api_key,
                            client=self._upstream_client,
                        )
                    except httpx.HTTPError as exc:
                        logger.error("pipeline: upstream transport error (param retry) — %s", exc)
                        retry_response = None
                    if retry_response is not None and retry_response.status_code < 400:
                        response = retry_response
                        normalized = retry_normalized
                        retried_ok = True
                    elif retry_response is not None:
                        error_body = await retry_response.aread()
                        upstream_status = retry_response.status_code
                        await retry_response.aclose()

            if not (retried_ok or retried_context_ok):
                fallback_model = self._config.provider.default_model.strip()
                if fallback_model and fallback_model != str(body.get("model", "")).strip() and _is_model_not_found_error(error_body):
                    retry_body = copy.deepcopy(body)
                    retry_body["model"] = fallback_model
                    retry_normalized = adapter.normalize_request(retry_body)
                    logger.warning(
                        "pipeline: upstream rejected model; retrying with default_model=%s (session=%s)",
                        fallback_model,
                        detection.session_key[:8],
                    )
                    try:
                        retry_response = await forward(
                            retry_normalized,
                            adapter,
                            self._config,
                            api_key=api_key,
                            client=self._upstream_client,
                        )
                    except httpx.HTTPError as exc:
                        logger.error("pipeline: upstream transport error (model retry) — %s", exc)
                        retry_response = None
                    if retry_response is not None and retry_response.status_code < 400:
                        response = retry_response
                        normalized = retry_normalized
                        retried_ok = True
                    elif retry_response is not None:
                        error_body = await retry_response.aread()
                        upstream_status = retry_response.status_code
                        await retry_response.aclose()

            if not (retried_ok or retried_context_ok):
                if _is_transient_upstream_error(upstream_status, error_body):
                    for delay_s in (0.5, 1.5):
                        await asyncio.sleep(delay_s)
                        logger.warning(
                            "pipeline: transient upstream error; retrying (status=%d, session=%s)",
                            upstream_status,
                            detection.session_key[:8],
                        )
                        try:
                            retry_response = await forward(
                                normalized,
                                adapter,
                                self._config,
                                api_key=api_key,
                                client=self._upstream_client,
                            )
                        except httpx.HTTPError as exc:
                            logger.error("pipeline: upstream transport error (transient retry) — %s", exc)
                            continue
                        if retry_response.status_code < 400:
                            response = retry_response
                            retried_ok = True
                            break
                        error_body = await retry_response.aread()
                        upstream_status = retry_response.status_code
                        await retry_response.aclose()
            if retried_ok or retried_context_ok:
                pass
            else:
                logger.error(
                    "pipeline: upstream returned %d — %s",
                    upstream_status,
                    error_body.decode()[:500],
                )
                if self._stats:
                    self._stats.record(
                        method="POST", path="/v1/chat/completions",
                        status_code=upstream_status,
                        latency_ms=round((_time.time() - t0) * 1000),
                        model=model, repair_applied=repair_applied,
                        api_key_available=has_key,
                    )
                fallback = _build_error_completion(
                    upstream_status=upstream_status,
                    upstream_body=error_body,
                    model=str(body.get("model", "")),
                )
                if requested_stream:
                    return StreamingResponse(
                        content=_single_chunk(_to_sse_error_stream(fallback)),
                        status_code=200,
                        media_type="text/event-stream",
                        headers={
                            "X-Accel-Buffering": "no",
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                return StreamingResponse(
                    content=_single_chunk(json.dumps(fallback).encode("utf-8")),
                    status_code=200,
                    media_type="application/json",
                )

        is_streaming = normalized.get("stream", False)

        if is_streaming:
            if self._stats:
                self._stats.record(
                    method="POST", path="/v1/chat/completions",
                    status_code=200,
                    latency_ms=round((_time.time() - t0) * 1000),
                    model=model, repair_applied=repair_applied,
                    api_key_available=has_key,
                )

            eval_cb = None
            if self._mission_store:
                def _cb(resp: dict[str, Any]) -> None:
                    self._evaluate_and_update_mission(resp)
                eval_cb = _cb

            return StreamingResponse(
                content=relay_stream(
                    response, detection.session_key, model, adapter, self._store,
                    evaluation_callback=eval_cb,
                ),
                status_code=200,
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        raw = await response.aread()
        await response.aclose()
        try:
            resp_json = json.loads(raw)
        except json.JSONDecodeError:
            resp_json = {}

        capture_from_assembled(resp_json, detection.session_key, adapter, self._store)

        # Stage: evaluate response against active mission
        self._evaluate_and_update_mission(resp_json)

        if self._stats:
            self._stats.record(
                method="POST", path="/v1/chat/completions",
                status_code=200,
                latency_ms=round((_time.time() - t0) * 1000),
                model=model, repair_applied=repair_applied,
                api_key_available=has_key,
            )

        return StreamingResponse(
            content=_single_chunk(raw),
            status_code=200,
            media_type="application/json",
        )

    async def _inject_visual_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self._vision_analyzer:
            return messages
        image_urls = self._vision_analyzer.resolve_message_image_urls(messages)
        if not image_urls:
            return messages

        prompt = _build_vision_prompt(messages)
        analysis = await self._vision_analyzer.analyze(image_urls, prompt)
        if not analysis:
            return messages

        visual_context = self._vision_analyzer.build_visual_context(analysis)
        return [
            {"role": "system", "content": visual_context},
            *messages,
        ]


    def _evaluate_and_update_mission(self, response_body: dict[str, Any]) -> None:
        """Evaluate response against the active mission and update state."""
        if self._mission_store is None:
            return
        mission = self._mission_store.get_active_mission()
        if mission is None:
            return
        result = evaluate_response(response_body, mission)
        mission.current_attempt = result.attempt
        if result.completed:
            mission.status = "completed"
            mission.completed_at = int(_time.time())
            mission.notes = result.feedback
            logger.info(
                "mission: '%s' COMPLETED on attempt %d",
                mission.mission_id, result.attempt,
            )
        elif result.failed:
            mission.status = "failed"
            mission.completed_at = int(_time.time())
            mission.notes = result.feedback
            logger.info(
                "mission: '%s' FAILED after %d attempts",
                mission.mission_id, result.attempt,
            )
        else:
            if result.feedback:
                mission.notes = result.feedback
        self._mission_store.save(mission)


async def _single_chunk(data: bytes) -> AsyncGenerator[bytes, None]:
    yield data


def _build_vision_prompt(messages: list[dict[str, Any]]) -> str:
    user_text: list[str] = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            user_text.append(content.strip())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    user_text.append(str(part.get("text", "")).strip())
    prompt = "Analyze this Android app screenshot for layout, design, usability, and visible text."
    if user_text:
        prompt += " User request context: " + " | ".join(filter(None, user_text[-3:]))
    return prompt


def _is_missing_reasoning_error(error_body: bytes) -> bool:
    try:
        text = error_body.decode("utf-8", errors="replace")
    except Exception:
        return False
    lower = text.lower()
    if "reasoning_content" not in lower:
        return False
    if "must be passed back" in lower:
        return True
    if "thinking mode" in lower and "passed back" in lower:
        return True
    return bool(re.search(r"reasoning_content.*passed back", lower))


def _strip_assistant_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"system", "developer", "user"}
    result = [copy.deepcopy(m) for m in messages if m.get("role") in allowed]
    if result:
        return result
    if messages:
        return [copy.deepcopy(messages[-1])]
    return []


def _is_context_length_error(error_body: bytes) -> bool:
    try:
        text = error_body.decode("utf-8", errors="replace")
    except Exception:
        return False
    lower = text.lower()
    if "maximum context length" in lower:
        return True
    if "context length" in lower and "requested" in lower and "tokens" in lower:
        return True
    return False


def _parse_context_length_limits(error_body: bytes) -> tuple[int | None, int | None]:
    try:
        text = error_body.decode("utf-8", errors="replace")
    except Exception:
        return None, None
    max_match = re.search(r"maximum context length is\s+(\d+)\s+tokens", text, flags=re.IGNORECASE)
    req_match = re.search(r"you requested\s+(\d+)\s+tokens", text, flags=re.IGNORECASE)
    max_tokens = int(max_match.group(1)) if max_match else None
    requested_tokens = int(req_match.group(1)) if req_match else None
    return max_tokens, requested_tokens


def _strip_reasoning_fields(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for m in messages:
        clone = copy.deepcopy(m)
        clone.pop("reasoning_content", None)
        clone.pop("thinking", None)
        cleaned.append(clone)
    return cleaned


def _strip_reasoning_mode_params(body: dict[str, Any]) -> None:
    body.pop("reasoning_effort", None)
    body.pop("reasoning", None)
    body.pop("thinking", None)


def _estimate_message_chars(messages: list[dict[str, Any]]) -> int:
    try:
        return len(json.dumps(messages, ensure_ascii=False))
    except Exception:
        total = 0
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, str):
                total += len(c)
        return total


def _trim_messages_to_char_budget(messages: list[dict[str, Any]], budget_chars: int) -> list[dict[str, Any]]:
    if budget_chars <= 0:
        return _strip_assistant_history(messages)

    system_like: list[dict[str, Any]] = []
    rest: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role in {"system", "developer"}:
            system_like.append(copy.deepcopy(m))
        else:
            rest.append(copy.deepcopy(m))

    result: list[dict[str, Any]] = []
    result.extend(system_like)

    baseline_chars = _estimate_message_chars(result)
    if baseline_chars >= budget_chars:
        return result[:]

    for m in reversed(rest):
        m.pop("reasoning_content", None)
        m.pop("thinking", None)
        m.pop("tool_calls", None)
        m.pop("function_call", None)
        tentative = result + [m]
        if _estimate_message_chars(tentative) > budget_chars:
            continue
        result.append(m)

    if len(result) == len(system_like):
        tail = _strip_assistant_history(messages)
        return system_like + tail

    return result


def _extract_openai_error_message(error_body: bytes) -> str:
    try:
        data = json.loads(error_body)
    except Exception:
        data = None
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()
        msg = data.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    try:
        text = error_body.decode("utf-8", errors="replace")
    except Exception:
        return ""
    return text.strip()


def _extract_unknown_parameter(error_body: bytes) -> str:
    msg = _extract_openai_error_message(error_body)
    if not msg:
        return ""
    patterns = [
        r"(unknown|unrecognized)\s+parameter\s*[:=]?\s*['\"]?([a-zA-Z0-9_.\[\]-]+)['\"]?",
        r"unknown\s+parameter\s*[:=]\s*['\"]?([a-zA-Z0-9_.\[\]-]+)['\"]?",
        r"unrecognized\s+request\s+argument\s+['\"]?([a-zA-Z0-9_.\[\]-]+)['\"]?",
        r"unknown\s+field\s+['\"]?([a-zA-Z0-9_.\[\]-]+)['\"]?",
    ]
    for pat in patterns:
        m = re.search(pat, msg, flags=re.IGNORECASE)
        if not m:
            continue
        candidate = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
        candidate = str(candidate).strip().strip("'\"")
        if candidate:
            return candidate
    return ""


def _delete_path_key(obj: dict[str, Any], path: str) -> None:
    if not path:
        return
    if "[" in path and "]" in path:
        base = path.split("[", 1)[0]
    else:
        base = path
    key = base.split(".", 1)[0].strip()
    if key in obj:
        obj.pop(key, None)


def _is_model_not_found_error(error_body: bytes) -> bool:
    msg = _extract_openai_error_message(error_body).lower()
    if not msg:
        return False
    if "model" in msg and ("not found" in msg or "does not exist" in msg):
        return True
    if "the model" in msg and "does not exist" in msg:
        return True
    return False


def _is_transient_upstream_error(status_code: int, error_body: bytes) -> bool:
    if status_code in {408, 429, 500, 502, 503, 504}:
        return True
    msg = _extract_openai_error_message(error_body).lower()
    if "rate limit" in msg or "too many requests" in msg:
        return True
    if "overloaded" in msg or "try again later" in msg:
        return True
    return False


def _build_error_completion(*, upstream_status: int, upstream_body: bytes, model: str) -> dict[str, Any]:
    msg = _extract_openai_error_message(upstream_body)
    if not msg:
        msg = f"Upstream error ({upstream_status})."
    content = f"Upstream error ({upstream_status}). {msg}"
    return {
        "id": "aslor-error",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _to_sse_error_stream(completion: dict[str, Any]) -> bytes:
    content = completion["choices"][0]["message"]["content"]
    created = int(_time.time())
    model = completion.get("model", "")
    chunk1 = {
        "id": completion.get("id", "aslor-error"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    chunk2 = {
        "id": completion.get("id", "aslor-error"),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
    }
    return (
        f"data: {json.dumps(chunk1)}\n\n"
        f"data: {json.dumps(chunk2)}\n\n"
        "data: [DONE]\n\n"
    ).encode("utf-8")
