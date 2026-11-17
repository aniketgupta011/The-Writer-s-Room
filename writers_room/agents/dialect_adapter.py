"""
Dynamic Dialect Adaptation — Milestone 3.

Modifies character voice profiles in real-time based on:
1. Cumulative emotional stress over the scene (scene beat progression and emotional state)
2. Who they are speaking to (authority vs. peer vs. subordinate code-switching)
3. Dynamic manual slider overrides from UI/API
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


# ── Peer / Authority Relationship Matrix ─────────────────────────────────────
# Maps (speaker_id -> listener_id) to the perceived social relationship:
# 'authority'   - Listener has higher institutional/social power
# 'peer'        - Listener is on equal social footing
# 'subordinate' - Listener is viewed as inferior/subordinate in status
RELATIONSHIP_MATRIX: Dict[str, Dict[str, str]] = {
    "merc_vance": {
        "dr_lena_soros": "authority",
        "tommy_k": "peer",
    },
    "dr_lena_soros": {
        "merc_vance": "subordinate",
        "tommy_k": "subordinate",
    },
    "tommy_k": {
        "merc_vance": "peer",
        "dr_lena_soros": "authority",
    },
}

# ── Authority Code-Switching Shifts ──────────────────────────────────────────
# How speaking to an authority, peer, or subordinate shifts a character's rules.
AUTHORITY_SHIFTS: Dict[str, Dict[str, Any]] = {
    "merc_vance": {
        "authority": {
            "max_words_per_sentence_delta": +3,     # Strains to form slightly longer, more careful sentences
            "contraction_rate_delta": -0.15,         # Dial back contractions slightly when dealing with authority
            "extra_forbidden": ["y'all", "'nuff"],  # Avoids most casual colloquialisms
            "extra_quirks": [
                "Addresses listener with strained deference ('Doc', 'Ma'am')",
                "Hedging maritime analogies with defensive caveats"
            ],
        },
        "peer": {
            "max_words_per_sentence_delta": -2,     # Terse, direct
            "contraction_rate_delta": +0.05,
            "extra_forbidden": [],
            "extra_quirks": ["Uses 'kid' or gruff dock terms as address markers"],
        },
        "subordinate": {
            "max_words_per_sentence_delta": -3,
            "contraction_rate_delta": +0.10,
            "extra_forbidden": [],
            "extra_quirks": ["Dismissive one-word grunts"],
        },
    },
    "dr_lena_soros": {
        "subordinate": {
            "max_words_per_sentence_delta": +4,     # Cascades of technical verbosity as a dominance display
            "contraction_rate_delta": -0.05,         # Even fewer contractions
            "extra_forbidden": ["yep", "nope", "guy", "folks"],
            "extra_quirks": [
                "Lecturing cadence; refers to listener as an untrained observer",
                "Overly enunciated clinical terminology"
            ],
        },
        "peer": {
            "max_words_per_sentence_delta": 0,
            "contraction_rate_delta": 0.0,
            "extra_forbidden": [],
            "extra_quirks": ["Colleague-level academic debate phrasing"],
        },
        "authority": {
            "max_words_per_sentence_delta": -3,
            "contraction_rate_delta": -0.05,
            "extra_forbidden": [],
            "extra_quirks": ["Formal institutional deference, citation-heavy defensiveness"],
        },
    },
    "tommy_k": {
        "authority": {
            "max_words_per_sentence_delta": +3,     # Panicked attempt to construct complete adult sentences
            "contraction_rate_delta": -0.15,
            # Tommy code-switches AWAY from street slang when intimidated by authority
            "extra_forbidden": ["bruh", "deadass", "no cap", "lowkey", "nah"],
            "extra_quirks": [
                "Nervously attempts technical/corporate buzzwords",
                "Stammers with dashes while trying to sound professional"
            ],
        },
        "peer": {
            "max_words_per_sentence_delta": -1,
            "contraction_rate_delta": +0.10,
            "extra_forbidden": [],
            "extra_quirks": [
                "Maximum slang density ('bruh', 'deadass')",
                "Unfiltered hype and pop-culture analogies"
            ],
        },
        "subordinate": {
            "max_words_per_sentence_delta": 0,
            "contraction_rate_delta": +0.05,
            "extra_forbidden": [],
            "extra_quirks": ["Condescending gamer/tech flexes"],
        },
    },
}

# ── Stress Escalation Curves ────────────────────────────────────────────────
# Stress progresses across scene beats and acute emotional states.
STRESS_THRESHOLDS = {
    "low": 0.30,     # Beat ratio < 30%
    "medium": 0.65,  # Beat ratio 30% - 65%
    "high": 1.00,    # Beat ratio > 65%
}

ACUTE_STRESS_EMOTIONS = {"angry", "aggressive", "desperate", "panicked"}

STRESS_MODIFIERS: Dict[str, Dict[str, Any]] = {
    "merc_vance": {
        "low":    {"max_words_delta": 0,  "contraction_delta": 0.0},
        "medium": {"max_words_delta": -2, "contraction_delta": +0.05},
        # Under high stress, Merc collapses into 3-5 word clipped fragments
        "high":   {"max_words_delta": -5, "contraction_delta": +0.10, "force_max_words_cap": 8},
    },
    "dr_lena_soros": {
        "low":    {"max_words_delta": 0,  "contraction_delta": 0.0},
        "medium": {"max_words_delta": -4, "contraction_delta": +0.05},
        # Under high stress, Soros's clinical mask cracks — sentences get shorter and icier
        "high":   {"max_words_delta": -8, "contraction_delta": +0.15, "force_max_words_cap": 14},
    },
    "tommy_k": {
        "low":    {"max_words_delta": 0,  "contraction_delta": 0.0},
        "medium": {"max_words_delta": +2, "contraction_delta": +0.05},
        # Under high stress, Tommy enters hyperactive panic mode (run-ons connected with dashes)
        "high":   {"max_words_delta": +4, "contraction_delta": -0.05, "add_quirk": "Rapid-fire repetitions when cornered"},
    },
}


def calculate_stress_level(
    beat_number: int,
    max_beats: int,
    emotional_state: str = "guarded",
) -> str:
    """Calculate scene stress level based on progression and current emotion."""
    if emotional_state.lower() in ACUTE_STRESS_EMOTIONS:
        return "high"
    
    ratio = beat_number / max(max_beats, 1)
    if ratio < STRESS_THRESHOLDS["low"]:
        return "low"
    elif ratio < STRESS_THRESHOLDS["medium"]:
        return "medium"
    return "high"


def get_relationship(speaker_id: str, listener_id: Optional[str]) -> str:
    """Return relationship classification (authority, peer, subordinate)."""
    if not listener_id or listener_id == speaker_id:
        return "peer"
    speaker_rels = RELATIONSHIP_MATRIX.get(speaker_id, {})
    return speaker_rels.get(listener_id, "peer")


def adapt_profile(
    base_profile: Dict[str, Any],
    speaker_id: str,
    listener_id: Optional[str] = None,
    beat_number: int = 1,
    max_beats: int = 10,
    emotional_state: str = "guarded",
    manual_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Dynamically modify a character's voice profile under stress and code-switching.

    Parameters
    ----------
    base_profile : dict
        Original character profile loaded from disk.
    speaker_id : str
        The speaking character ID.
    listener_id : str, optional
        Target listener in the scene.
    beat_number : int
        Current scene beat.
    max_beats : int
        Total planned scene beats.
    emotional_state : str
        Director's assigned emotional state for this beat.
    manual_overrides : dict, optional
        User-applied dialect sliders from UI.

    Returns
    -------
    dict
        A deep copy of the profile with dynamically adapted rules.
    """
    profile = deepcopy(base_profile)
    dialect = profile.setdefault("dialect_rules", {})

    base_max_words = int(dialect.get("max_words_per_sentence", 14))
    raw_contraction = dialect.get("contraction_rate", 0.5)
    if isinstance(raw_contraction, str):
        contraction_map = {"high": 0.85, "medium": 0.55, "low": 0.10}
        base_contraction = contraction_map.get(raw_contraction.lower(), 0.50)
    else:
        base_contraction = float(raw_contraction)

    max_words_delta = 0
    contraction_delta = 0.0
    extra_forbidden: List[str] = []
    extra_quirks: List[str] = []

    # 1. Stress Progression
    stress = calculate_stress_level(beat_number, max_beats, emotional_state)
    stress_conf = STRESS_MODIFIERS.get(speaker_id, {}).get(stress, {})
    max_words_delta += stress_conf.get("max_words_delta", 0)
    contraction_delta += stress_conf.get("contraction_delta", 0.0)
    if "add_quirk" in stress_conf:
        extra_quirks.append(stress_conf["add_quirk"])

    # 2. Relationship-based Code-Switching
    relationship = get_relationship(speaker_id, listener_id)
    rel_shifts = AUTHORITY_SHIFTS.get(speaker_id, {}).get(relationship, {})
    max_words_delta += rel_shifts.get("max_words_per_sentence_delta", 0)
    contraction_delta += rel_shifts.get("contraction_rate_delta", 0.0)
    extra_forbidden.extend(rel_shifts.get("extra_forbidden", []))
    extra_quirks.extend(rel_shifts.get("extra_quirks", []))

    # Calculate final word limit & contraction rate
    final_max_words = max(3, base_max_words + max_words_delta)
    if "force_max_words_cap" in stress_conf:
        final_max_words = min(final_max_words, stress_conf["force_max_words_cap"])

    final_contraction = max(0.0, min(1.0, base_contraction + contraction_delta))

    dialect["max_words_per_sentence"] = final_max_words
    dialect["contraction_rate"] = round(final_contraction, 2)

    if extra_forbidden:
        current_forbidden = list(dialect.get("forbidden_lexicon", []))
        for word in extra_forbidden:
            if word not in current_forbidden:
                current_forbidden.append(word)
        dialect["forbidden_lexicon"] = current_forbidden

    if extra_quirks:
        current_quirks = list(dialect.get("syntax_quirks", []))
        for quirk in extra_quirks:
            if quirk not in current_quirks:
                current_quirks.append(quirk)
        dialect["syntax_quirks"] = current_quirks

    # 3. Manual Slider Overrides (if any)
    if manual_overrides:
        if "sentence_length" in manual_overrides:
            offset = int(manual_overrides["sentence_length"])
            dialect["max_words_per_sentence"] = max(3, dialect["max_words_per_sentence"] + offset)
        if "slang_density" in manual_overrides:
            pct = float(manual_overrides["slang_density"]) / 100.0
            dialect["contraction_rate"] = round(pct, 2)
        if "formality" in manual_overrides:
            formality_pct = float(manual_overrides["formality"]) / 100.0
            # Higher formality reduces contractions and increases sentence length
            dialect["contraction_rate"] = round(max(0.0, dialect["contraction_rate"] - (formality_pct * 0.3)), 2)

    # 4. Attach Adaptation Metadata for Transparency & UI
    profile["_adaptation_meta"] = {
        "stress_level": stress,
        "relationship_to_listener": relationship,
        "listener_id": listener_id,
        "original_max_words": base_max_words,
        "adapted_max_words": dialect["max_words_per_sentence"],
        "original_contraction_rate": round(base_contraction, 2),
        "adapted_contraction_rate": dialect["contraction_rate"],
        "code_switched": bool(extra_forbidden or extra_quirks),
    }

    return profile
