"""Tests for StreamRelayAgent internals."""

from __future__ import annotations

import json
import pytest
from aslor.agents.relay import (
    _parse_sse_chunk,
    _build_synthetic_response,
    _split_sse_events,
    _normalize_sse_event_for_client,
)


class TestParseSseChunk:
    def _make_chunk(self, delta: dict) -> bytes:
        data = {"choices": [{"delta": delta}]}
        return f"data: {json.dumps(data)}\n\n".encode()

    def test_accumulates_reasoning_content(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk = self._make_chunk({"reasoning_content": "think ", "content": ""})
        _parse_sse_chunk(chunk, acc_rc, acc_ct, "")
        assert acc_rc == ["think "]
        assert acc_ct == []

    def test_accumulates_content(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk = self._make_chunk({"reasoning_content": "", "content": "hello"})
        _parse_sse_chunk(chunk, acc_rc, acc_ct, "")
        assert acc_ct == ["hello"]

    def test_ignores_done_marker(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk = b"data: [DONE]\n\n"
        _parse_sse_chunk(chunk, acc_rc, acc_ct, "")
        assert acc_rc == []
        assert acc_ct == []

    def test_ignores_malformed_json(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk = b"data: {bad json}\n\n"
        _parse_sse_chunk(chunk, acc_rc, acc_ct, "")
        assert acc_rc == []

    def test_ignores_non_data_lines(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk = b"event: ping\n\n"
        _parse_sse_chunk(chunk, acc_rc, acc_ct, "")
        assert acc_rc == []

    def test_returns_empty_partial_for_complete_chunk(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk = self._make_chunk({"content": "ok"})
        remainder = _parse_sse_chunk(chunk, acc_rc, acc_ct, "")
        assert remainder == ""

    def test_buffers_partial_line_across_chunks(self):
        """When a chunk ends mid-line, the partial content is returned for buffering."""
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        # First chunk: partial — missing the closing `}` and newline
        chunk1 = b'data: {"choices":[{"delta":{"content":"hel'
        remainder = _parse_sse_chunk(chunk1, acc_rc, acc_ct, "")
        assert remainder == 'data: {"choices":[{"delta":{"content":"hel'

        # Second chunk: continuation — prepend the remainder
        chunk2 = b'lo"}}]}\n'
        remainder2 = _parse_sse_chunk(chunk2, acc_rc, acc_ct, remainder)
        assert remainder2 == ""
        assert acc_ct == ["hello"]

    def test_buffers_partial_data_value_across_chunks(self):
        """Data value split across chunks with partial line prepended."""
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk1 = b'data: {"choices":[{"delta":{"content":"wor'
        remainder = _parse_sse_chunk(chunk1, acc_rc, acc_ct, "")
        assert remainder != ""

        chunk2 = b'ld"}}]}\n\n'
        _parse_sse_chunk(chunk2, acc_rc, acc_ct, remainder)
        assert acc_ct == ["world"]

    def test_multiple_events_in_one_chunk(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        events = b'data: {"choices":[{"delta":{"content":"a"}}]}\n\ndata: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        _parse_sse_chunk(events, acc_rc, acc_ct, "")
        assert acc_ct == ["a", "b"]

    def test_reasoning_content_across_chunks(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk1 = b'data: {"choices":[{"delta":{"reasoning_content":"I need to '
        remainder = _parse_sse_chunk(chunk1, acc_rc, acc_ct, "")
        chunk2 = b'think about this"}}]}\n\n'
        _parse_sse_chunk(chunk2, acc_rc, acc_ct, remainder)
        assert "".join(acc_rc) == "I need to think about this"

    def test_accumulates_reasoning_from_delta_reasoning_field(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk = self._make_chunk({"reasoning": "think", "content": ""})
        _parse_sse_chunk(chunk, acc_rc, acc_ct, "")
        assert "".join(acc_rc) == "think"

    def test_accumulates_content_from_list_parts(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        chunk = self._make_chunk({"content": [{"type": "text", "text": "hi "}, {"type": "text", "text": "there"}]})
        _parse_sse_chunk(chunk, acc_rc, acc_ct, "")
        assert "".join(acc_ct) == "hi there"

    def test_accumulates_responses_api_reasoning_event(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        event = b'data: {"type":"response.reasoning_text.delta","delta":"thinking"}\n\n'
        _parse_sse_chunk(event, acc_rc, acc_ct, "")
        assert "".join(acc_rc) == "thinking"

    def test_accumulates_responses_api_output_text_event(self):
        acc_rc: list[str] = []
        acc_ct: list[str] = []
        event = b'data: {"type":"response.output_text.delta","delta":"answer"}\n\n'
        _parse_sse_chunk(event, acc_rc, acc_ct, "")
        assert "".join(acc_ct) == "answer"


class TestSplitSseEvents:
    def test_splits_multiple_events_lf(self):
        payload = b"data: 1\n\ndata: 2\n\n"
        events, remainder = _split_sse_events(payload)
        assert remainder == b""
        assert events == [b"data: 1\n\n", b"data: 2\n\n"]

    def test_splits_multiple_events_crlf(self):
        payload = b"data: 1\r\n\r\ndata: 2\r\n\r\n"
        events, remainder = _split_sse_events(payload)
        assert remainder == b""
        assert events == [b"data: 1\r\n\r\n", b"data: 2\r\n\r\n"]

    def test_keeps_partial_remainder(self):
        payload = b"data: 1\n\ndata: 2"
        events, remainder = _split_sse_events(payload)
        assert events == [b"data: 1\n\n"]
        assert remainder == b"data: 2"


class TestBuildSyntheticResponse:
    def test_builds_valid_structure(self):
        resp = _build_synthetic_response("thoughts", "answer", "reasoning_content")
        assert resp["choices"][0]["message"]["reasoning_content"] == "thoughts"
        assert resp["choices"][0]["message"]["content"] == "answer"

    def test_uses_custom_field(self):
        resp = _build_synthetic_response("t", "a", "thinking")
        assert "thinking" in resp["choices"][0]["message"]


class TestNormalizeSseEventForClient:
    def test_converts_responses_output_text_delta_to_chat_completion_chunks(self):
        event = b'data: {"type":"response.output_text.delta","delta":"Hello"}\n\n'
        out, convert_mode, sent_role, saw_done = _normalize_sse_event_for_client(
            event=event,
            convert_responses_stream=False,
            sent_role=False,
            saw_done=False,
            created=123,
            stream_id="aslor-test",
            model="m",
        )
        assert convert_mode is True
        assert sent_role is True
        assert saw_done is False
        assert len(out) == 2
        assert b'"object": "chat.completion.chunk"' in out[0]
        assert b'"role": "assistant"' in out[0]
        assert b'"content": "Hello"' in out[1]

    def test_passes_through_chat_completions_events_unchanged(self):
        event = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        out, convert_mode, sent_role, saw_done = _normalize_sse_event_for_client(
            event=event,
            convert_responses_stream=False,
            sent_role=False,
            saw_done=False,
            created=123,
            stream_id="aslor-test",
            model="m",
        )
        assert out == [event]
        assert convert_mode is False
        assert sent_role is False
        assert saw_done is False
