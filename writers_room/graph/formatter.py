"""Screenplay / Fountain formatter for the Writers' Room pipeline.

Converts the dialogue list and scene metadata into proper Fountain format
(https://fountain.io/syntax).
"""

from __future__ import annotations

from typing import Any


def _normalise_location(location: str) -> str:
    """Ensure the location slug follows Fountain conventions.

    Fountain scene headings start with INT., EXT., or INT./EXT. followed by
    the location in uppercase and optionally a time-of-day suffix.
    If the caller already provided a properly-prefixed location, leave it
    alone; otherwise default to ``INT. <LOCATION> - DAY``.
    """
    loc = location.strip()
    upper = loc.upper()
    if upper.startswith(("INT.", "EXT.", "INT./EXT.")):
        return upper
    return f"INT. {upper} - DAY"


def _character_display_name(character_id: str, profiles: dict[str, dict]) -> str:
    """Resolve a character_id to its display_name (uppercased for Fountain).

    Falls back to the character_id itself, title-cased and uppercased.
    """
    profile = profiles.get(character_id, {})
    name = profile.get("display_name", character_id.replace("_", " ").title())
    return name.upper()


def format_screenplay(scene_state: dict[str, Any]) -> str:
    """Convert a completed scene state dict into Fountain-formatted text.

    Parameters
    ----------
    scene_state:
        Must contain at minimum ``location`` (str), ``dialogue``
        (list of DialogueLine dicts), and ``character_profiles``
        (dict mapping character_id → profile dict).  ``premise`` is used
        as an optional action line at the top of the scene.

    Returns
    -------
    str
        The scene rendered in Fountain markup.
    """
    lines: list[str] = []

    # ── Scene heading ────────────────────────────────────────────────
    location = scene_state.get("location", "UNKNOWN LOCATION")
    lines.append(_normalise_location(location))
    lines.append("")

    # Optional premise / action line
    premise = scene_state.get("premise", "")
    if premise:
        lines.append(premise)
        lines.append("")

    # ── Dialogue lines ───────────────────────────────────────────────
    profiles: dict[str, dict] = scene_state.get("character_profiles", {})
    dialogue: list[dict[str, Any]] = scene_state.get("dialogue", [])

    for entry in dialogue:
        character_id: str = entry.get("character_id", "unknown")
        spoken_text: str = entry.get("spoken_text", "")
        stage_direction: str = entry.get("stage_direction", "")

        # Character name (uppercase, centred in Fountain readers)
        lines.append(_character_display_name(character_id, profiles))

        # Parenthetical / stage direction
        if stage_direction:
            # Fountain parentheticals are wrapped in parens on their own line
            direction = stage_direction.strip()
            if not direction.startswith("("):
                direction = f"({direction})"
            if not direction.endswith(")"):
                direction = f"{direction})"
            lines.append(direction)

        # Spoken dialogue
        lines.append(spoken_text)

        # Blank line separates dialogue blocks
        lines.append("")

    # Strip any trailing blank lines, then ensure a single final newline
    text = "\n".join(lines).rstrip()
    return text + "\n"
