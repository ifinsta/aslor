"""Mission context injection into chat messages."""

from __future__ import annotations

import logging
from typing import Any

from aslor.missions.models import MissionState, SkillDefinition

logger = logging.getLogger(__name__)


def inject_mission_context(
    messages: list[dict[str, Any]],
    mission: MissionState | None,
    skills: list[SkillDefinition],
) -> list[dict[str, Any]]:
    """Inject mission goal and active skills as a prepended system message.

    Safe passthrough: if *mission* is None, returns *messages* unchanged.
    """
    if mission is None:
        return messages

    attempt_display = min(mission.current_attempt + 1, max(mission.max_attempts, 1))
    ctx_parts: list[str] = [
        "=== MISSION ===",
        f"Goal: {mission.title}",
        f"Description: {mission.description}",
        f"Success Criteria: {mission.success_criteria}",
        f"Attempt {attempt_display} of {mission.max_attempts}",
    ]

    if skills:
        ctx_parts.append("")
        ctx_parts.append("=== ACTIVE SKILLS ===")
        for skill in skills:
            ctx_parts.append(f"[{skill.title}] {skill.system_prompt.strip()}")

    if mission.current_attempt > 0 and mission.notes:
        ctx_parts.append("")
        ctx_parts.append("=== CORRECTIVE GUIDANCE (from previous attempt) ===")
        ctx_parts.append(mission.notes)

    context_message: dict[str, Any] = {
        "role": "system",
        "content": "\n".join(ctx_parts),
    }

    result = list(messages)
    result.insert(0, context_message)
    logger.info(
        "mission: injected context for '%s' (attempt %d/%d, skills=%d)",
        mission.mission_id,
        attempt_display,
        mission.max_attempts,
        len(skills),
    )
    return result
