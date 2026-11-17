"""LangGraph state-machine for the Writers' Room scene-generation pipeline.

The graph implements a cyclic Director → Character → Linter → Critic pipeline
with conditional retry logic and a finalize step.

Usage
-----
>>> from writers_room.graph.scene_graph import build_scene_graph
>>> graph = build_scene_graph()
>>> result = await graph.ainvoke(initial_state)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from writers_room.agents.character import CharacterAgent
from writers_room.agents.director import DirectorAgent
from writers_room.graph.formatter import format_screenplay
from writers_room.linter.voice_linter import lint_dialogue

logger = logging.getLogger(__name__)


# ── State schema ─────────────────────────────────────────────────────


class WritersRoomState(TypedDict):
    """Full mutable state threaded through every graph node."""

    scene_id: str
    premise: str
    location: str
    active_characters: list[str]
    character_profiles: dict[str, dict]          # character_id → profile
    beats: list[dict]                            # SceneBeat dicts
    dialogue: list[dict]                         # DialogueLine dicts
    current_beat: dict | None                    # Current SceneBeat
    current_line: dict | None                    # Current generated line
    current_adapted_profile: dict | None         # Dynamically adapted profile for current beat
    lint_result: dict | None                     # LintResult
    critic_verdict: dict | None                  # CriticVerdict
    retry_count: int                             # Current retry count
    max_retries: int                             # Max retries (default 2)
    max_beats: int                               # Total beats to generate
    errors: list[str]                            # Error log
    is_complete: bool                            # Scene completion flag

    # Runtime objects (not serialised) — injected before invoke
    _director: Any                               # DirectorAgent instance
    _character_pool: Any                         # dict[str, CharacterAgent]
    _event_queue: Any                            # asyncio.Queue | None


# ── Helpers ──────────────────────────────────────────────────────────


def _to_dict(obj: Any) -> dict:
    """Convert a Pydantic model, dataclass, or plain object to a dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)
    return vars(obj)


def _emit(state: WritersRoomState, event_type: str, data: dict) -> None:
    """Push a JSON-serialisable event onto the WebSocket queue (if present)."""
    queue: asyncio.Queue | None = state.get("_event_queue")
    if queue is not None:
        try:
            queue.put_nowait({"type": event_type, "data": data})
        except Exception:  # noqa: BLE001
            pass


async def _run_critic(line_text: str, profile: dict, golden_lines: list[str]) -> dict:
    """Call the LLM critic, returning a CriticVerdict dict.

    Imported lazily so the graph module doesn't hard-fail if the linter
    agent hasn't been deployed yet.
    """
    try:
        from writers_room.linter.critic import critique_dialogue  # type: ignore[import-untyped]
        verdict = await critique_dialogue(line_text, profile, golden_lines)
        if isinstance(verdict, dict):
            return verdict
        return _to_dict(verdict)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Critic call failed: %s — auto-passing", exc)
        return {"passed": True, "feedback": f"Critic unavailable ({exc})", "similarity_score": 0.0}


# ── Graph node functions ─────────────────────────────────────────────


async def director_node(state: WritersRoomState) -> dict:
    """Ask the DirectorAgent for the next dramatic beat."""
    director: DirectorAgent = state["_director"]
    scene_state_snapshot = {
        "scene_id": state["scene_id"],
        "premise": state["premise"],
        "location": state["location"],
        "active_characters": state["active_characters"],
        "beats": list(state["beats"]),
        "dialogue": list(state["dialogue"]),
    }

    try:
        beat = await director.generate_beat(scene_state_snapshot)
        beat_dict = _to_dict(beat)
        beat_dict.setdefault("beat_number", len(state["beats"]) + 1)
    except Exception as exc:  # noqa: BLE001
        logger.error("Director failed: %s", exc)
        # Fabricate a fallback beat that round-robins speakers
        idx = len(state["beats"]) % max(len(state["active_characters"]), 1)
        speaker = state["active_characters"][idx] if state["active_characters"] else "unknown"
        beat_dict = {
            "beat_number": len(state["beats"]) + 1,
            "next_speaker": speaker,
            "tactical_intent": "Continue the scene naturally.",
            "emotional_state": "guarded",
            "physical_action": "",
        }
        return {
            "beats": state["beats"] + [beat_dict],
            "current_beat": beat_dict,
            "errors": state["errors"] + [f"Director error: {exc}"],
        }

    _emit(state, "beat", beat_dict)
    return {
        "beats": state["beats"] + [beat_dict],
        "current_beat": beat_dict,
    }


async def character_node(state: WritersRoomState) -> dict:
    """Generate a dialogue line for the current beat's speaker."""
    beat = state["current_beat"]
    if beat is None:
        return {"errors": state["errors"] + ["character_node: no current_beat"]}

    speaker_id: str = beat["next_speaker"]
    pool: dict[str, CharacterAgent] = state.get("_character_pool", {})
    agent = pool.get(speaker_id)

    if agent is None:
        logger.error("No CharacterAgent for %s", speaker_id)
        return {"errors": state["errors"] + [f"No agent for character '{speaker_id}'"]}

    # Build recent dialogue context (last 3 spoken lines)
    recent: list[str] = [
        d.get("spoken_text", "") for d in state["dialogue"][-3:]
    ]

    # Resolve listener for relationship-based code-switching
    active = state.get("active_characters", [])
    other_chars = [c for c in active if c != speaker_id]
    last_speaker = None
    for d in reversed(state.get("dialogue", [])):
        if d.get("character_id") != speaker_id:
            last_speaker = d.get("character_id")
            break
    listener_id = last_speaker or (other_chars[0] if other_chars else None)

    # Dynamic dialect adaptation under stress & relationship code-switching
    base_profile = state.get("character_profiles", {}).get(speaker_id, {})
    try:
        from writers_room.agents.dialect_adapter import adapt_profile
        adapted_profile = adapt_profile(
            base_profile=base_profile,
            speaker_id=speaker_id,
            listener_id=listener_id,
            beat_number=beat.get("beat_number", len(state["beats"])),
            max_beats=state.get("max_beats", 10),
            emotional_state=beat.get("emotional_state", "guarded"),
            manual_overrides=base_profile.get("_dynamic_modifiers"),
        )
        if "_adaptation_meta" in adapted_profile:
            _emit(state, "adaptation", adapted_profile["_adaptation_meta"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Dialect adaptation failed: %s — falling back to base profile", exc)
        adapted_profile = base_profile

    try:
        # Reconstruct a lightweight SceneBeat for the agent
        from writers_room.data.schemas import SceneBeat  # type: ignore[import-untyped]
        scene_beat = SceneBeat(**beat)
        line = await agent.generate_line(scene_beat, recent, adapted_profile=adapted_profile)
        line_dict = _to_dict(line)
    except Exception as exc:  # noqa: BLE001
        logger.error("Character agent (%s) failed: %s", speaker_id, exc)
        line_dict = {
            "character_id": speaker_id,
            "spoken_text": "...",
            "stage_direction": "",
            "beat_number": beat.get("beat_number", 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return {
            "current_line": line_dict,
            "current_adapted_profile": adapted_profile,
            "errors": state["errors"] + [f"Character ({speaker_id}) error: {exc}"],
        }

    line_dict.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    _emit(state, "line", line_dict)
    return {"current_line": line_dict, "current_adapted_profile": adapted_profile}


async def deterministic_lint_node(state: WritersRoomState) -> dict:
    """Run the deterministic voice-linter on the current line."""
    line = state.get("current_line")
    if line is None:
        return {
            "lint_result": {"passed": True, "violations": [], "stage": "deterministic"},
        }

    speaker_id: str = line.get("character_id", "")
    profile: dict = (
        state.get("current_adapted_profile")
        or state.get("character_profiles", {}).get(speaker_id, {})
    )
    spoken_text: str = line.get("spoken_text", "")

    try:
        result = lint_dialogue(spoken_text, profile)
        result_dict = _to_dict(result)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Lint call failed: %s — auto-passing", exc)
        result_dict = {"passed": True, "violations": [f"Lint error: {exc}"], "stage": "deterministic"}

    if result_dict.get("passed"):
        _emit(state, "lint_pass", result_dict)
    else:
        _emit(state, "lint_fail", result_dict)
    return {"lint_result": result_dict}


async def critic_node(state: WritersRoomState) -> dict:
    """Run the LLM style-critic on the current line."""
    line = state.get("current_line")
    if line is None:
        return {
            "critic_verdict": {"passed": True, "feedback": "No line to critique.", "similarity_score": 0.0},
        }

    speaker_id: str = line.get("character_id", "")
    profile: dict = state.get("character_profiles", {}).get(speaker_id, {})
    spoken_text: str = line.get("spoken_text", "")

    # Retrieve golden lines from the character agent's vector store
    golden_lines: list[str] = []
    pool: dict[str, CharacterAgent] = state.get("_character_pool", {})
    agent = pool.get(speaker_id)
    if agent is not None and hasattr(agent, "_vector_store") and agent._vector_store is not None:
        beat = state.get("current_beat") or {}
        emotion = beat.get("emotional_state", "guarded")
        try:
            golden_lines = agent._vector_store.query_golden_lines(
                character_id=speaker_id, emotion=emotion, n_results=3,
            )
        except Exception:  # noqa: BLE001
            pass

    verdict = await _run_critic(spoken_text, profile, golden_lines)

    if verdict.get("passed"):
        _emit(state, "critic_pass", verdict)
    else:
        _emit(state, "critic_fail", verdict)
    return {"critic_verdict": verdict}


async def accept_line_node(state: WritersRoomState) -> dict:
    """Accept the current line and append it to the dialogue history."""
    line = state.get("current_line")
    if line is None:
        return {}

    new_dialogue = list(state["dialogue"]) + [line]
    _emit(state, "line_accepted", line)
    return {
        "dialogue": new_dialogue,
        "current_line": None,
        "current_adapted_profile": None,
        "lint_result": None,
        "critic_verdict": None,
        "retry_count": 0,
    }


async def reject_line_node(state: WritersRoomState) -> dict:
    """Reject the current line: bump retry count and log violations."""
    violations: list[str] = []
    lint_result = state.get("lint_result")
    critic_verdict = state.get("critic_verdict")

    if lint_result and not lint_result.get("passed"):
        violations.extend(lint_result.get("violations", []))
    if critic_verdict and not critic_verdict.get("passed"):
        violations.append(critic_verdict.get("feedback", "Critic rejected"))

    new_retry = state["retry_count"] + 1
    log_entry = (
        f"Retry {new_retry}/{state['max_retries']} for beat "
        f"{state.get('current_beat', {}).get('beat_number', '?')}: "
        + "; ".join(violations)
    )
    _emit(state, "retry", {"retry_count": new_retry, "violations": violations})
    return {
        "retry_count": new_retry,
        "errors": state["errors"] + [log_entry],
        "lint_result": None,
        "critic_verdict": None,
    }


async def finalize_node(state: WritersRoomState) -> dict:
    """Mark the scene complete and format the screenplay."""
    screenplay = format_screenplay({
        "location": state["location"],
        "premise": state["premise"],
        "dialogue": state["dialogue"],
        "character_profiles": state.get("character_profiles", {}),
    })
    _emit(state, "complete", {"screenplay": screenplay})
    return {
        "is_complete": True,
        "screenplay": screenplay,
    }


# ── Conditional edge routers ─────────────────────────────────────────


def after_lint(state: WritersRoomState) -> str:
    """Route after deterministic lint."""
    lint = state.get("lint_result")
    if lint and lint.get("passed"):
        return "critic_node"
    return "reject_line_node"


def after_critic(state: WritersRoomState) -> str:
    """Route after LLM critic."""
    verdict = state.get("critic_verdict")
    if verdict and verdict.get("passed"):
        return "accept_line_node"
    return "reject_line_node"


def after_reject(state: WritersRoomState) -> str:
    """Route after rejection: retry or accept with warning."""
    if state["retry_count"] < state["max_retries"]:
        return "character_node"
    # Exhausted retries — accept with a warning
    return "accept_line_node"


def after_accept(state: WritersRoomState) -> str:
    """Route after acceptance: more beats or finalize."""
    if len(state["dialogue"]) < state["max_beats"]:
        return "director_node"
    return "finalize_node"


# ── Graph builder ────────────────────────────────────────────────────


def build_scene_graph() -> Any:
    """Construct and compile the Writers' Room LangGraph state machine.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to be invoked with
        ``await graph.ainvoke(initial_state)``.
    """
    workflow = StateGraph(WritersRoomState)

    # Register nodes
    workflow.add_node("director_node", director_node)
    workflow.add_node("character_node", character_node)
    workflow.add_node("deterministic_lint_node", deterministic_lint_node)
    workflow.add_node("critic_node", critic_node)
    workflow.add_node("accept_line_node", accept_line_node)
    workflow.add_node("reject_line_node", reject_line_node)
    workflow.add_node("finalize_node", finalize_node)

    # Entry point
    workflow.set_entry_point("director_node")

    # Fixed edges
    workflow.add_edge("director_node", "character_node")
    workflow.add_edge("character_node", "deterministic_lint_node")

    # Conditional edges
    workflow.add_conditional_edges(
        "deterministic_lint_node",
        after_lint,
        {"critic_node": "critic_node", "reject_line_node": "reject_line_node"},
    )
    workflow.add_conditional_edges(
        "critic_node",
        after_critic,
        {"accept_line_node": "accept_line_node", "reject_line_node": "reject_line_node"},
    )
    workflow.add_conditional_edges(
        "reject_line_node",
        after_reject,
        {"character_node": "character_node", "accept_line_node": "accept_line_node"},
    )
    workflow.add_conditional_edges(
        "accept_line_node",
        after_accept,
        {"director_node": "director_node", "finalize_node": "finalize_node"},
    )

    # Finalize → END
    workflow.add_edge("finalize_node", END)

    return workflow.compile()


# ── Convenience: build initial state dict ────────────────────────────


def make_initial_state(
    *,
    premise: str,
    location: str,
    active_characters: list[str],
    character_profiles: dict[str, dict],
    max_beats: int = 6,
    max_retries: int = 2,
    director: DirectorAgent | None = None,
    character_pool: dict[str, CharacterAgent] | None = None,
    event_queue: asyncio.Queue | None = None,
) -> WritersRoomState:
    """Build a ready-to-invoke initial state dict."""
    return WritersRoomState(
        scene_id=str(uuid.uuid4()),
        premise=premise,
        location=location,
        active_characters=active_characters,
        character_profiles=character_profiles,
        beats=[],
        dialogue=[],
        current_beat=None,
        current_line=None,
        current_adapted_profile=None,
        lint_result=None,
        critic_verdict=None,
        retry_count=0,
        max_retries=max_retries,
        max_beats=max_beats,
        errors=[],
        is_complete=False,
        _director=director,
        _character_pool=character_pool or {},
        _event_queue=event_queue,
    )
