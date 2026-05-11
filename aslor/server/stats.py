"""In-memory request statistics and rotating history buffer."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestRecord:
    timestamp: float
    method: str
    path: str
    status_code: int
    latency_ms: int
    model: str | None = None
    repair_applied: bool = False
    api_key_available: bool = False


@dataclass
class StatsSnapshot:
    total_requests: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    repair_count: int = 0
    recent_requests: list[dict[str, Any]] = field(default_factory=list)
    uptime_seconds: float = 0.0


class StatsTracker:
    """Thread-safe in-memory tracker for request metrics."""

    def __init__(self, max_history: int = 100) -> None:
        self._max = max_history
        self._total = 0
        self._success = 0
        self._errors = 0
        self._repairs = 0
        self._latency_samples: list[int] = []
        self._history: deque[RequestRecord] = deque(maxlen=max_history)
        self._start_time = time.time()

    def record(
        self,
        *,
        method: str = "POST",
        path: str = "/v1/chat/completions",
        status_code: int = 200,
        latency_ms: int = 0,
        model: str | None = None,
        repair_applied: bool = False,
        api_key_available: bool = False,
    ) -> None:
        entry = RequestRecord(
            timestamp=time.time(),
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=latency_ms,
            model=model or "",
            repair_applied=repair_applied,
            api_key_available=api_key_available,
        )
        self._total += 1
        if 200 <= status_code < 400:
            self._success += 1
        else:
            self._errors += 1
        if repair_applied:
            self._repairs += 1
        self._latency_samples.append(latency_ms)
        # keep at most 200 samples for average
        if len(self._latency_samples) > 200:
            self._latency_samples = self._latency_samples[-200:]
        self._history.append(entry)

    def snapshot(self) -> StatsSnapshot:
        avg = (
            sum(self._latency_samples) / len(self._latency_samples)
            if self._latency_samples
            else 0.0
        )
        return StatsSnapshot(
            total_requests=self._total,
            success_count=self._success,
            error_count=self._errors,
            avg_latency_ms=round(avg, 1),
            repair_count=self._repairs,
            recent_requests=[
                {
                    "timestamp": r.timestamp,
                    "method": r.method,
                    "path": r.path,
                    "status": r.status_code,
                    "latency_ms": r.latency_ms,
                    "model": r.model or "",
                    "repair": r.repair_applied,
                }
                for r in reversed(self._history)
            ],
            uptime_seconds=round(time.time() - self._start_time),
        )

    @property
    def total(self) -> int:
        return self._total
