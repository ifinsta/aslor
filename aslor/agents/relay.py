"""StreamRelayAgent — streams SSE chunks from upstream back to Android Studio.

For streaming responses, this agent:
1. Iterates over the raw SSE byte stream from the upstream provider.
2. Accumulates ``reasoning_content`` delta fragments.
3. Yields each chunk unchanged to the FastAPI StreamingResponse.
4. After the stream ends, assembles a synthetic response dict so that
   ResponseCaptureAgent can persist the reasoning state.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, AsyncGenerator

import httpx

from aslor.agents.capture import capture_from_assembled
from aslor.providers.base import ProviderAdapter
from aslor.reasoning.state import ReasoningStateStore

logger = logging.getLogger(__name__)


async def relay_stream(
    response: httpx.Response,
    session_id: str,
    model: str,
    adapter: ProviderAdapter,
    store: ReasoningStateStore,
    evaluation_callback: Callable[[dict[str, Any]], None] | None = None,
) -> AsyncGenerator[bytes, None]:
    """Yield raw SSE bytes while accumulating state for caching.

    If *evaluation_callback* is provided, it is called with the assembled
    response after the stream ends, so mission evaluation can run on
    streaming responses.
    """
    accumulated_reasoning: list[str] = []
    accumulated_content: list[str] = []
    partial_line = ""
    buffer = b""
    convert_responses_stream = False
    sent_role = False
    saw_done = False
    created = int(time.time())
    stream_id = f"aslor-{session_id[:12]}"
    model_name = str(model or "")

    try:
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            buffer += chunk
            events, buffer = _split_sse_events(buffer)
            for event in events:
                partial_line = _parse_sse_chunk(
                    event,
                    accumulated_reasoning,
                    accumulated_content,
                    partial_line,
                )
                out_events, convert_responses_stream, sent_role, saw_done = _normalize_sse_event_for_client(
                    event=event,
                    convert_responses_stream=convert_responses_stream,
                    sent_role=sent_role,
                    saw_done=saw_done,
                    created=created,
                    stream_id=stream_id,
                    model=model_name,
                )
                for out in out_events:
                    yield out
        if buffer:
            _parse_sse_chunk(
                buffer,
                accumulated_reasoning,
                accumulated_content,
                partial_line,
            )
    finally:
        if convert_responses_stream and not saw_done:
            yield b"data: [DONE]\n\n"
        await response.aclose()
        reasoning_text = "".join(accumulated_reasoning)
        content_text = "".join(accumulated_content)
        if reasoning_text and adapter.reasoning_field():
            synthetic = _build_synthetic_response(
                reasoning_text=reasoning_text,
                content_text=content_text,
                field=adapter.reasoning_field(),
            )
            capture_from_assembled(synthetic, session_id, adapter, store)
            if evaluation_callback:
                evaluation_callback(synthetic)
        elif evaluation_callback and content_text:
            evaluation_callback({
                "choices": [{"message": {"role": "assistant", "content": content_text}}],
            })


def _split_sse_events(buffer: bytes) -> tuple[list[bytes], bytes]:
    events: list[bytes] = []
    while True:
        idx_lf = buffer.find(b"\n\n")
        idx_crlf = buffer.find(b"\r\n\r\n")
        if idx_lf == -1 and idx_crlf == -1:
            break
        if idx_crlf != -1 and (idx_lf == -1 or idx_crlf < idx_lf):
            end = idx_crlf + 4
        else:
            end = idx_lf + 2
        events.append(buffer[:end])
        buffer = buffer[end:]
    return events, buffer


def _normalize_sse_event_for_client(
    *,
    event: bytes,
    convert_responses_stream: bool,
    sent_role: bool,
    saw_done: bool,
    created: int,
    stream_id: str,
    model: str,
) -> tuple[list[bytes], bool, bool, bool]:
    if b"data:" not in event:
        return [event], convert_responses_stream, sent_role, saw_done
    if b"data: [DONE]" in event:
        return [event], convert_responses_stream, sent_role, True

    parsed_any = False
    out: list[bytes] = []
    for line in event.splitlines():
        if not line.startswith(b"data:"):
            continue
        data_str = line[5:].strip()
        if not data_str:
            continue
        try:
            data = json.loads(data_str.decode("utf-8", errors="replace"))
        except Exception:
            continue
        parsed_any = True

        if not convert_responses_stream:
            if isinstance(data, dict) and isinstance(data.get("type"), str) and "choices" not in data:
                convert_responses_stream = True

        if not convert_responses_stream:
            out.append(event)
            break

        event_type = data.get("type") if isinstance(data, dict) else None
        if isinstance(event_type, str):
            if not sent_role:
                out.append(_format_sse_data(_build_role_chunk(stream_id=stream_id, created=created, model=model)))
                sent_role = True
            if "reasoning" in event_type:
                text = _extract_text(data.get("delta"))
                if text:
                    out.append(_format_sse_data(_build_delta_chunk(
                        stream_id=stream_id, created=created, model=model, delta={"reasoning_content": text},
                    )))
                continue
            if "output_text" in event_type or event_type.endswith(".text.delta") or event_type.endswith(".output_text.delta"):
                text = _extract_text(data.get("delta"))
                if text:
                    out.append(_format_sse_data(_build_delta_chunk(
                        stream_id=stream_id, created=created, model=model, delta={"content": text},
                    )))
                continue
            if event_type.endswith(".completed") or event_type.endswith(".done"):
                continue

    if not parsed_any:
        return [event], convert_responses_stream, sent_role, saw_done
    if not out and not convert_responses_stream:
        return [event], convert_responses_stream, sent_role, saw_done
    if not out and convert_responses_stream:
        return [], convert_responses_stream, sent_role, saw_done
    return out, convert_responses_stream, sent_role, saw_done


def _format_sse_data(obj: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


def _build_role_chunk(*, stream_id: str, created: int, model: str) -> dict[str, Any]:
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }


def _build_delta_chunk(*, stream_id: str, created: int, model: str, delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


def _parse_sse_chunk(
    chunk: bytes,
    accumulated_reasoning: list[str],
    accumulated_content: list[str],
    partial_line: str,
) -> str:
    """Parse SSE bytes, accumulating ``reasoning_content`` and ``content`` deltas.

    Returns any trailing partial line that should be prepended to the next chunk.
    """
    text = partial_line + chunk.decode("utf-8", errors="replace")
    lines = text.splitlines()

    # If the raw text doesn't end with a newline, the last line is partial.
    if not text.endswith("\n"):
        partial_line = lines.pop() if lines else ""
    else:
        partial_line = ""

    for line in lines:
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("type"), str):
            event_type = data.get("type")
            delta = data.get("delta")
            text = _extract_text(delta)
            if text:
                if "reasoning" in event_type:
                    accumulated_reasoning.append(text)
                elif "output_text" in event_type or event_type.endswith(".text.delta") or event_type.endswith(".output_text.delta"):
                    accumulated_content.append(text)
            continue

        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not isinstance(choices, list) or not choices:
            continue
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        delta = choice0.get("delta")
        if not isinstance(delta, dict):
            delta = choice0.get("message", {}) if isinstance(choice0.get("message"), dict) else {}

        rc = _extract_text(delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking"))
        ct = _extract_text(delta.get("content"))
        if rc:
            accumulated_reasoning.append(rc)
        if ct:
            accumulated_content.append(ct)

    return partial_line


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
        return "".join(parts) if parts else ""
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text", ""))
        if "text" in value:
            return str(value.get("text", ""))
        return ""
    return ""


def _build_synthetic_response(reasoning_text: str, content_text: str, field: str) -> dict[str, Any]:
    """Build a minimal response dict for capture_from_assembled."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": content_text,
        field: reasoning_text,
    }
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
    }
