"""Response evaluation for mission completion and failure detection.

Uses keyword matching on the assistant response to detect progress signals.
This is intentionally simple rather than AI-based to keep latency near zero.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from aslor.missions.models import MissionState

logger = logging.getLogger(__name__)

_COMPLETION_KEYWORDS = [
    "mission complete",
    "goal achieved",
    "task accomplished",
    "objective met",
    "all requirements satisfied",
    "implementation complete",
    "all tests pass",
    "ready for production",
]

_FAILURE_KEYWORDS = [
    "cannot complete",
    "unable to accomplish",
    "cannot be done",
    "this is impossible",
    "i cannot achieve",
    "blocked indefinitely",
    "no viable solution",
]


@dataclass
class EvaluationResult:
    completed: bool = False
    failed: bool = False
    attempt: int = 0
    feedback: str = ""


def evaluate_response(
    response_body: dict[str, Any],
    mission: MissionState,
) -> EvaluationResult:
    """Analyze an assistant response for mission completion or failure.

    Returns an ``EvaluationResult`` the caller uses to update mission state.
    The *attempt* field is always incremented by 1 from current.
    """
    choices = response_body.get("choices", [])
    if not choices:
        return EvaluationResult(attempt=mission.current_attempt)

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str):
        content = str(content) if content else ""

    content_lower = content.lower()
    next_attempt = mission.current_attempt + 1

    # Check for completion signals
    for kw in _COMPLETION_KEYWORDS:
        if kw in content_lower:
            logger.info(
                "mission: completion keyword '%s' matched for '%s'",
                kw, mission.mission_id,
            )
            return EvaluationResult(
                completed=True,
                attempt=next_attempt,
                feedback=f"Response matched completion signal: '{kw}'.",
            )

    # Check if success criteria words all appear in response
    criteria_words = _significant_words(mission.success_criteria)
    if criteria_words and len(criteria_words) >= 2:
        if all(w in content_lower for w in criteria_words):
            logger.info(
                "mission: all success_criteria words matched for '%s'",
                mission.mission_id,
            )
            return EvaluationResult(
                completed=True,
                attempt=next_attempt,
                feedback="All success criteria keywords found in response.",
            )

    # Check for failure signals
    for kw in _FAILURE_KEYWORDS:
        if kw in content_lower:
            exceeded = next_attempt >= mission.max_attempts
            logger.info(
                "mission: failure keyword '%s' matched for '%s' (attempt %d/%d -> %s)",
                kw, mission.mission_id, next_attempt, mission.max_attempts,
                "FAILED" if exceeded else "retrying",
            )
            return EvaluationResult(
                failed=exceeded,
                attempt=next_attempt,
                feedback=(
                    f"Max attempts ({mission.max_attempts}) exhausted. "
                    "Mission marked as failed. Create a new mission or reset this one."
                    if exceeded
                    else "Response indicated difficulty. Incrementing attempt counter "
                    "and injecting corrective guidance on next request."
                ),
            )

    # No terminal signal — mission continues
    exceeded = next_attempt >= mission.max_attempts
    if exceeded:
        return EvaluationResult(
            failed=True,
            attempt=next_attempt,
            feedback=(
                f"Max attempts ({mission.max_attempts}) exhausted. "
                "Mission marked as failed. Create a new mission or reset this one."
            ),
        )
    return EvaluationResult(attempt=next_attempt)


def _significant_words(text: str, max_words: int = 5) -> list[str]:
    """Extract significant lowercased words (4+ chars) from criteria text."""
    if not text:
        return []
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    return words[:max_words]
