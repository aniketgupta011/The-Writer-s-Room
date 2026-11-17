"""
LLM Style Critic — Stage B of the two-stage dialogue critic.

Uses the Gemini API (google-genai) to compare a generated dialogue line
against golden reference lines and score its voice authenticity.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class CriticVerdict:
    """Result of the LLM-based style critique."""
    passed: bool
    feedback: str
    similarity_score: float

    def to_dict(self) -> dict:
        return asdict(self)


from ..config import CRITIC_MODEL, FALLBACK_MODELS

# The lightweight model used for critic calls.
_MODEL = CRITIC_MODEL

_CRITIC_PROMPT_TEMPLATE = """\
You are a dialect authenticity critic. Compare this generated line against the character's canonical voice samples.

Generated line: "{line}"
Character archetype: {archetype}

Golden reference lines:
1. "{golden_1}"
2. "{golden_2}"
3. "{golden_3}"

Evaluate the generated line on these dimensions:
- Therapy-speak: Does it use therapeutic language ("I feel", "I hear you", "It's okay to…") that the character wouldn't?
- Overly balanced clauses: Does it hedge both sides unnaturally ("while X, also Y")?
- Unnatural hedging: Does it soften too much ("perhaps", "it might be worth considering")?
- Vocabulary mismatch: Does it use words outside this character's register?
- Rhythm mismatch: Does the sentence length, cadence, and flow diverge from the golden lines?

Score authenticity 0.0–1.0. Flag specific phrases that feel like AI-generated text rather than this character's natural voice.
Respond in JSON: {{"passed": bool, "score": float, "feedback": "..."}}
Pass threshold: 0.7
"""


async def critique_dialogue(
    line: str,
    profile: Dict[str, Any],
    golden_lines: List[str],
) -> CriticVerdict:
    """Run the LLM style critic on *line* using *golden_lines* as reference.

    Parameters
    ----------
    line : str
        The generated dialogue line to evaluate.
    profile : dict
        The character voice profile (must contain at least ``archetype``).
    golden_lines : list[str]
        Exactly 3 golden reference lines for the character.

    Returns
    -------
    CriticVerdict
        The critic's verdict including pass/fail, feedback, and score.
    """
    if len(golden_lines) < 3:
        raise ValueError("critique_dialogue requires exactly 3 golden lines")

    archetype = profile.get("archetype", profile.get("display_name", "unknown"))

    prompt = _CRITIC_PROMPT_TEMPLATE.format(
        line=line,
        archetype=archetype,
        golden_1=golden_lines[0],
        golden_2=golden_lines[1],
        golden_3=golden_lines[2],
    )

    # Import google.genai lazily so the module loads even without the SDK.
    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "The google-genai package is required for the LLM critic. "
            "Install it with: pip install google-genai"
        )

    client = genai.Client()

    models_to_try = [_MODEL] + [m for m in FALLBACK_MODELS if m != _MODEL]
    last_exc = None
    response = None
    for try_model in models_to_try:
        try:
            response = await client.aio.models.generate_content(
                model=try_model,
                contents=prompt,
            )
            break
        except Exception as e:
            last_exc = e
            continue

    if response is None:
        if last_exc:
            raise last_exc
        raise RuntimeError("Critic failed to generate response")

    # Parse the JSON response from the model.
    raw_text = response.text.strip()

    # The model may wrap JSON in a ```json code fence — strip it.
    if raw_text.startswith("```"):
        # Remove opening fence line and closing fence.
        lines = raw_text.split("\n")
        # Drop first line (```json) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw_text = "\n".join(lines).strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return CriticVerdict(
            passed=False,
            feedback=f"Failed to parse critic response: {raw_text[:200]}",
            similarity_score=0.0,
        )

    score = float(data.get("score", 0.0))
    passed = data.get("passed", score >= 0.7)
    feedback = data.get("feedback", "")

    return CriticVerdict(
        passed=bool(passed),
        feedback=feedback,
        similarity_score=score,
    )
