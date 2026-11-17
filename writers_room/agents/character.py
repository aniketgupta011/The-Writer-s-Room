"""Character Agent Factory — produces in-character dialogue lines."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai

from ..config import CHARACTER_MODEL, FALLBACK_MODELS
from ..data.schemas import DialogueLine, SceneBeat

# ── Prompt template ──────────────────────────────────────────────────

_CHARACTER_SYSTEM_TEMPLATE = """\
You are exclusively playing {display_name}. Respond ONLY with your character's spoken dialogue and immediate physical action in JSON:
{{"spoken_text": "...", "stage_direction": "..."}}

STRICT VOICE CONSTRAINTS:
- Max sentence length: {max_words_per_sentence} words.
- Syntax rules: {syntax_quirks}
- NEVER use these words or modern AI phrasing: {forbidden_lexicon}
- Current emotional state: {emotional_state} (Behavior rule: {subtext_behavior_for_state})

CANONICAL VOICE EXAMPLES (Match this exact rhythm and cadence):
1. "{golden_line_1}"
2. "{golden_line_2}"
3. "{golden_line_3}"

RECENT DIALOGUE (last 3 lines for context):
{recent_dialogue_formatted}

SCENE CONTEXT:
The Director instructs you to: {tactical_intent}
Your physical action: {physical_action}"""


def _extract_json(text: str) -> dict[str, Any]:
    """Try ``json.loads`` first; fall back to regex extraction."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


class CharacterAgent:
    """An agent that speaks in a single character's voice.

    Parameters
    ----------
    character_id : str
        Unique slug matching the voice-profile filename.
    voice_profile : dict[str, Any]
        Parsed voice-profile JSON (matches :class:`CharacterVoiceProfile`).
    vector_store : module | object
        An object or module exposing ``query_golden_lines(character_id, emotion, n_results)``.
    model : str
        Gemini model name (default ``gemini-2.5-flash``).
    """

    def __init__(
        self,
        character_id: str,
        voice_profile: dict[str, Any],
        vector_store: Any,
        model: str | None = None,
    ) -> None:
        self.character_id = character_id
        self.voice_profile = voice_profile
        self._vector_store = vector_store
        self._model = model or CHARACTER_MODEL
        self._client = genai.Client()
        self.update_profile(voice_profile)

    def update_profile(self, voice_profile: dict[str, Any]) -> None:
        """Update cached voice constraints from a modified profile."""
        self.voice_profile = voice_profile
        self._display_name: str = voice_profile.get("display_name", self.character_id)
        dialect: dict[str, Any] = voice_profile.get("dialect_rules", {})
        self._max_words: int = dialect.get("max_words_per_sentence", 20)
        self._syntax_quirks: str = ", ".join(dialect.get("syntax_quirks", []))
        self._forbidden: str = ", ".join(dialect.get("forbidden_lexicon", []))
        self._subtext: dict[str, str] = voice_profile.get("subtext_behavior", {})

    def _subtext_for_emotion(self, emotional_state: str) -> str:
        """Look up the subtext behaviour string for a given emotion."""
        key = f"when_{emotional_state}"
        if key in self._subtext:
            return self._subtext[key]
        # Try partial match (e.g., "aggressive" → "when_angry")
        emotion_map = {
            "aggressive": "when_angry",
            "angry": "when_angry",
            "guarded": "when_guarded",
            "desperate": "when_desperate",
            "mocking": "when_mocking",
            "vulnerable": "when_desperate",
            "sarcastic": "when_mocking",
            "analytical": "when_guarded",
        }
        mapped = emotion_map.get(emotional_state.lower(), "when_guarded")
        return self._subtext.get(mapped, "Stay in character.")

    async def generate_line(
        self,
        beat: SceneBeat,
        recent_dialogue: list[str],
        adapted_profile: dict[str, Any] | None = None,
    ) -> DialogueLine:
        """Generate a single dialogue line for this character.

        Parameters
        ----------
        beat : SceneBeat
            The current dramatic beat from the Director.
        recent_dialogue : list[str]
            The last few dialogue lines as formatted strings.
        adapted_profile : dict, optional
            A dynamically adapted voice profile (e.g. from dialect_adapter).
        """
        # Resolve active voice rules (adapted or base)
        active_profile = adapted_profile or self.voice_profile
        dialect = active_profile.get("dialect_rules", {})
        max_words = dialect.get("max_words_per_sentence", self._max_words)
        syntax_quirks = ", ".join(dialect.get("syntax_quirks", [])) or self._syntax_quirks
        forbidden = ", ".join(dialect.get("forbidden_lexicon", [])) or self._forbidden
        subtext_text = self._subtext_for_emotion(beat.emotional_state)

        # Fetch golden-line examples from the vector store
        golden_lines: list[str] = self._vector_store.query_golden_lines(
            character_id=self.character_id,
            emotion=beat.emotional_state,
            n_results=3,
        )
        # Pad to exactly 3 if fewer returned
        while len(golden_lines) < 3:
            golden_lines.append("(no example available)")

        # Format recent dialogue
        recent_formatted = "\n".join(recent_dialogue[-3:]) if recent_dialogue else "(none)"

        # Build the system prompt
        system_prompt = _CHARACTER_SYSTEM_TEMPLATE.format(
            display_name=self._display_name,
            max_words_per_sentence=max_words,
            syntax_quirks=syntax_quirks,
            forbidden_lexicon=forbidden,
            emotional_state=beat.emotional_state,
            subtext_behavior_for_state=subtext_text,
            golden_line_1=golden_lines[0],
            golden_line_2=golden_lines[1],
            golden_line_3=golden_lines[2],
            recent_dialogue_formatted=recent_formatted,
            tactical_intent=beat.tactical_intent,
            physical_action=beat.physical_action or "(no specific action)",
        )

        user_message = (
            f"Deliver your next line as {self._display_name}. "
            f"Emotion: {beat.emotional_state}. "
            f"Intent: {beat.tactical_intent}"
        )

        models_to_try = [self._model] + [m for m in FALLBACK_MODELS if m != self._model]
        last_exc = None
        response = None
        for try_model in models_to_try:
            try:
                response = await self._client.aio.models.generate_content(
                    model=try_model,
                    contents=user_message,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.85,
                    ),
                )
                break
            except Exception as e:
                last_exc = e
                continue

        if response is None:
            if last_exc:
                raise last_exc
            raise RuntimeError(f"CharacterAgent({self.character_id}) failed to generate line")

        raw_text = response.text or ""
        data = _extract_json(raw_text)

        return DialogueLine(
            character_id=self.character_id,
            spoken_text=data.get("spoken_text", raw_text.strip()),
            stage_direction=data.get("stage_direction", ""),
            beat_number=beat.beat_number,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ── Factory ──────────────────────────────────────────────────────

    @classmethod
    def create_agent_pool(
        cls,
        profiles_dir: str,
        vector_store: Any,
        model: str | None = None,
    ) -> dict[str, "CharacterAgent"]:
        """Load every voice-profile JSON in *profiles_dir* and return a dict
        mapping ``character_id → CharacterAgent``.
        """
        pool: dict[str, CharacterAgent] = {}
        profiles_path = Path(profiles_dir)

        for filepath in sorted(profiles_path.glob("*.json")):
            with open(filepath, "r", encoding="utf-8") as fh:
                profile = json.load(fh)
            cid: str = profile.get("character_id", filepath.stem)
            pool[cid] = cls(
                character_id=cid,
                voice_profile=profile,
                vector_store=vector_store,
                model=model or CHARACTER_MODEL,
            )

        return pool
