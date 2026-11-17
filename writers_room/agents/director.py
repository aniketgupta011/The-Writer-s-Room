"""Director Agent — generates scene beats without writing dialogue."""

from __future__ import annotations

import json
import re
from typing import Any

from google import genai

from ..config import DIRECTOR_MODEL, FALLBACK_MODELS
from ..data.schemas import SceneBeat

_DIRECTOR_SYSTEM_PROMPT = """\
You are the Scene Director. Do NOT write dialogue.
Given the scene premise and conversation history, output the next dramatic beat as JSON:
{
  "next_speaker": "character_id",  
  "tactical_intent": "What the character is trying to do to the other person (1 sentence)",
  "emotional_state": "guarded|aggressive|desperate|mocking|analytical|vulnerable|sarcastic",
  "physical_action": "Brief stage direction grounded in the environment"
}
Rules:
- Rotate speakers naturally. Never let one character dominate.
- Escalate tension through the scene arc.
- Vary emotional states across beats.
- Physical actions must reference the specific location.
- tactical_intent should create dramatic conflict, not agreement."""


def _extract_json(text: str) -> dict[str, Any]:
    """Try ``json.loads`` first; fall back to regex extraction."""
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Regex fallback — grab the first { … }
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


class DirectorAgent:
    """Generates dramatic beats for a scene using Gemini.

    Parameters
    ----------
    model : str
        The Gemini model name (default ``gemini-2.5-pro``).
    """

    def __init__(self, model: str | None = None) -> None:
        self._model = model or DIRECTOR_MODEL
        self._client = genai.Client()

    async def generate_beat(self, scene_state: dict[str, Any]) -> SceneBeat:
        """Produce the next :class:`SceneBeat` for the given scene state.

        *scene_state* should contain at minimum:
        - ``premise``  (str)
        - ``location`` (str)
        - ``active_characters`` (list[str])
        - ``beats``    (list[dict]) — prior beats
        - ``dialogue`` (list[dict]) — prior dialogue lines
        """
        # Build the beat number
        prior_beats: list[dict[str, Any]] = scene_state.get("beats", [])
        beat_number = len(prior_beats) + 1

        # Build user-message describing the current scene
        dialogue_history = scene_state.get("dialogue", [])
        history_text = "\n".join(
            f"  {d.get('character_id', '???')}: {d.get('spoken_text', '')}"
            for d in dialogue_history[-10:]  # last 10 lines for context
        ) or "(scene start — no dialogue yet)"

        active = ", ".join(scene_state.get("active_characters", []))
        user_message = (
            f"Scene premise: {scene_state.get('premise', '')}\n"
            f"Location: {scene_state.get('location', '')}\n"
            f"Active characters: {active}\n"
            f"Beat number: {beat_number}\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"Generate beat #{beat_number}."
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
                        system_instruction=_DIRECTOR_SYSTEM_PROMPT,
                        temperature=0.9,
                    ),
                )
                break
            except Exception as e:
                last_exc = e
                continue

        if response is None:
            if last_exc:
                raise last_exc
            raise RuntimeError("Director failed to generate response")

        raw_text = response.text or ""
        data = _extract_json(raw_text)

        # Inject beat_number (the LLM doesn't return it)
        data["beat_number"] = beat_number

        # Map "aggressive" → a supported emotional state if needed
        valid_emotions = {
            "guarded",
            "aggressive",
            "desperate",
            "mocking",
            "analytical",
            "vulnerable",
            "sarcastic",
            "angry",
        }
        if data.get("emotional_state", "").lower() not in valid_emotions:
            data["emotional_state"] = "guarded"  # safe default

        return SceneBeat.model_validate(data)
