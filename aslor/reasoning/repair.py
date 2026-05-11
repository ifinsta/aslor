"""ReasoningRepairAgent.

Finds assistant messages that are missing their required reasoning-state field
and re-injects the cached state from prior turns.

Safe passthrough guarantee: if no cached state is found, the messages are
returned unchanged. This agent never blocks a request.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from aslor.agents.capture import assistant_message_key, assistant_message_key_variants
from aslor.providers.base import ProviderAdapter
from aslor.reasoning.state import ReasoningStateStore

logger = logging.getLogger(__name__)


def repair_messages(
    messages: list[dict[str, Any]],
    session_id: str,
    adapter: ProviderAdapter,
    store: ReasoningStateStore,
) -> list[dict[str, Any]]:
    """Return messages with missing reasoning state re-injected.

    If no state is cached, or the model does not use reasoning, returns the
    original messages list unchanged.
    """
    field = adapter.reasoning_field()
    if not field:
        return messages

    assistant_indexes = [idx for idx, msg in enumerate(messages) if msg.get("role") == "assistant"]
    if not assistant_indexes:
        logger.debug("repair: no assistant missing %s - skipping", field)
        return messages

    latest_state = store.load(session_id)
    history = store.load_history(session_id)
    history_by_key: dict[str, dict[str, Any]] = {}
    history_states_in_order: list[dict[str, Any]] = []
    for item in history:
        key = item.get("assistant_key")
        state = item.get("state")
        if isinstance(key, str) and isinstance(state, dict):
            history_by_key[key] = state
            history_states_in_order.append(state)

    requested_assistant_keys = [
        key
        for idx in assistant_indexes
        if isinstance(messages[idx], dict)
        for key in assistant_message_key_variants(messages[idx])
    ]
    global_history_by_key = store.find_message_states(
        requested_assistant_keys,
        exclude_session_id=session_id,
    )

    if latest_state is None and not history and not global_history_by_key:
        logger.warning(
            "repair: missing %s detected but NO cached state for session %s "
            "(model may have switched, or cache was cleared, or first-turn capture failed) "
            "- forwarding unchanged",
            field,
            session_id,
        )
        return messages

    repaired: list[dict[str, Any]] = []
    unmatched_missing_indexes: list[int] = []
    assistant_ordinals_by_index: dict[int, int] = {}
    assistant_ordinal = -1
    exact_matches = 0
    global_exact_matches = 0

    for idx, msg in enumerate(messages):
        clone = copy.deepcopy(msg)
        if clone.get("role") != "assistant":
            repaired.append(clone)
            continue
        assistant_ordinal += 1
        assistant_ordinals_by_index[idx] = assistant_ordinal

        matched_state = None
        global_state = None
        for key in assistant_message_key_variants(clone):
            matched_state = history_by_key.get(key)
            if matched_state:
                break
        if not matched_state:
            for key in assistant_message_key_variants(clone):
                global_state = global_history_by_key.get(key)
                if global_state:
                    break
        if matched_state:
            repaired.append(_apply_message_state(adapter, clone, matched_state))
            exact_matches += 1
            continue

        if global_state:
            repaired.append(_apply_message_state(adapter, clone, global_state))
            global_exact_matches += 1
            continue

        if clone.get(field):
            repaired.append(clone)
            continue

        repaired.append(clone)
        unmatched_missing_indexes.append(idx)

    positional_repairs = 0
    still_unmatched_indexes: list[int] = []
    if exact_matches > 0:
        for idx in unmatched_missing_indexes:
            ordinal = assistant_ordinals_by_index.get(idx, -1)
            if 0 <= ordinal < len(history_states_in_order):
                repaired[idx] = _apply_message_state(adapter, repaired[idx], history_states_in_order[ordinal])
                positional_repairs += 1
            else:
                still_unmatched_indexes.append(idx)
    else:
        still_unmatched_indexes = list(unmatched_missing_indexes)

    if still_unmatched_indexes and latest_state is not None:
        for fallback_index in still_unmatched_indexes:
            repaired[fallback_index] = _apply_message_state(adapter, repaired[fallback_index], latest_state)
        logger.info(
            "repair: repaired %s using exact session matches=%d global exact matches=%d positional fallback=%d latest fallback=%d (session=%s)",
            field,
            exact_matches,
            global_exact_matches,
            positional_repairs,
            len(still_unmatched_indexes),
            session_id,
        )
        return repaired

    if exact_matches or global_exact_matches or positional_repairs:
        logger.info(
            "repair: repaired %s from history (session=%s, exact=%d, global_exact=%d, positional=%d)",
            field,
            session_id,
            exact_matches,
            global_exact_matches,
            positional_repairs,
        )
    return repaired


def _apply_message_state(
    adapter: ProviderAdapter,
    message: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    clone = copy.deepcopy(message)
    for key, value in state.items():
        clone[key] = copy.deepcopy(value)
    injected = adapter.inject_reasoning_state([clone], state)
    return injected[0] if injected else clone


def _has_assistant_missing_field(messages: list[dict[str, Any]], field: str) -> bool:
    """Return True if any assistant message is missing the reasoning field."""
    for msg in messages:
        if msg.get("role") == "assistant" and not msg.get(field):
            return True
    return False
