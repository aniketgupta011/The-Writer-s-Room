"""
Tests for Milestone 3: Dynamic Dialect Adaptation and Real-Time State Modifiers.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from writers_room.agents.dialect_adapter import (
    adapt_profile,
    calculate_stress_level,
    get_relationship,
    RELATIONSHIP_MATRIX,
)

VOICE_PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "voice_profiles"


@pytest.fixture
def merc_profile():
    with open(VOICE_PROFILES_DIR / "merc_vance.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def soros_profile():
    with open(VOICE_PROFILES_DIR / "dr_lena_soros.json", "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def tommy_profile():
    with open(VOICE_PROFILES_DIR / "tommy_k.json", "r", encoding="utf-8") as f:
        return json.load(f)


class TestStressCalculation:
    """Test stress escalation curves."""

    def test_progression_curve(self):
        # Beat 1 of 10 -> low stress
        assert calculate_stress_level(1, 10, "guarded") == "low"
        # Beat 5 of 10 -> medium stress
        assert calculate_stress_level(5, 10, "guarded") == "medium"
        # Beat 8 of 10 -> high stress
        assert calculate_stress_level(8, 10, "guarded") == "high"

    def test_acute_emotion_triggers_high_stress(self):
        # Even at beat 1, angry/desperate emotions trigger high stress
        assert calculate_stress_level(1, 10, "angry") == "high"
        assert calculate_stress_level(1, 10, "desperate") == "high"
        assert calculate_stress_level(1, 10, "aggressive") == "high"


class TestRelationshipMatrix:
    """Test relationship lookup between characters."""

    def test_authority_and_peer_relationships(self):
        assert get_relationship("tommy_k", "dr_lena_soros") == "authority"
        assert get_relationship("tommy_k", "merc_vance") == "peer"
        assert get_relationship("merc_vance", "dr_lena_soros") == "authority"
        assert get_relationship("merc_vance", "tommy_k") == "peer"
        assert get_relationship("dr_lena_soros", "merc_vance") == "subordinate"
        assert get_relationship("dr_lena_soros", "tommy_k") == "subordinate"

    def test_self_or_missing_defaults_to_peer(self):
        assert get_relationship("tommy_k", "tommy_k") == "peer"
        assert get_relationship("tommy_k", None) == "peer"
        assert get_relationship("tommy_k", "unknown_char") == "peer"


class TestTommyCodeSwitching:
    """Test Tommy K code-switches when addressing authority vs peer."""

    def test_tommy_drops_slang_with_authority(self, tommy_profile):
        # When Tommy speaks to Dr. Soros (authority)
        adapted = adapt_profile(
            base_profile=tommy_profile,
            speaker_id="tommy_k",
            listener_id="dr_lena_soros",
            beat_number=2,
            max_beats=10,
            emotional_state="guarded",
        )

        forbidden = adapted["dialect_rules"]["forbidden_lexicon"]
        # Street slang markers should now be forbidden
        for slang in ["bruh", "deadass", "no cap"]:
            assert slang in forbidden, f"Expected '{slang}' to be forbidden when speaking to authority"

        assert adapted["_adaptation_meta"]["relationship_to_listener"] == "authority"
        assert adapted["_adaptation_meta"]["code_switched"] is True

    def test_tommy_keeps_slang_with_peer(self, tommy_profile):
        # When Tommy speaks to Merc (peer)
        adapted = adapt_profile(
            base_profile=tommy_profile,
            speaker_id="tommy_k",
            listener_id="merc_vance",
            beat_number=2,
            max_beats=10,
            emotional_state="guarded",
        )

        forbidden = adapted["dialect_rules"]["forbidden_lexicon"]
        # Street slang should NOT be forbidden with a peer
        assert "bruh" not in forbidden
        assert "deadass" not in forbidden
        assert adapted["_adaptation_meta"]["relationship_to_listener"] == "peer"


class TestMercStressAdaptation:
    """Test Merc Vance shifts speech patterns under stress."""

    def test_merc_high_stress_clipped_sentences(self, merc_profile):
        # Under high stress (beat 9 of 10)
        adapted = adapt_profile(
            base_profile=merc_profile,
            speaker_id="merc_vance",
            listener_id="tommy_k",
            beat_number=9,
            max_beats=10,
            emotional_state="angry",
        )

        max_words = adapted["dialect_rules"]["max_words_per_sentence"]
        # Merc's base limit is 14; under high stress it must be capped at ≤ 8 words
        assert max_words <= 8, f"Expected capped max words <= 8, got {max_words}"
        assert adapted["_adaptation_meta"]["stress_level"] == "high"

    def test_merc_authority_shift(self, merc_profile):
        # Merc speaking to Dr. Soros (authority) in early scene
        adapted = adapt_profile(
            base_profile=merc_profile,
            speaker_id="merc_vance",
            listener_id="dr_lena_soros",
            beat_number=1,
            max_beats=10,
            emotional_state="guarded",
        )

        # Contraction rate should be lower (more guarded/formal)
        orig_rate = merc_profile["dialect_rules"]["contraction_rate"]
        adapted_rate = adapted["dialect_rules"]["contraction_rate"]
        assert adapted_rate < orig_rate
        # Extra quirks for strained deference
        quirks = " ".join(adapted["dialect_rules"]["syntax_quirks"])
        assert "deference" in quirks.lower() or "Doc" in quirks


class TestSorosAdaptation:
    """Test Dr. Lena Soros linguistic shifts."""

    def test_soros_subordinate_verbosity(self, soros_profile):
        # Speaking down to Merc Vance
        adapted = adapt_profile(
            base_profile=soros_profile,
            speaker_id="dr_lena_soros",
            listener_id="merc_vance",
            beat_number=1,
            max_beats=10,
            emotional_state="analytical",
        )

        orig_words = soros_profile["dialect_rules"]["max_words_per_sentence"]
        adapted_words = adapted["dialect_rules"]["max_words_per_sentence"]
        assert adapted_words > orig_words, "Expected increased sentence length for dominance display"

    def test_soros_high_stress_mask_cracks(self, soros_profile):
        # Under acute crisis/desperation
        adapted = adapt_profile(
            base_profile=soros_profile,
            speaker_id="dr_lena_soros",
            listener_id="merc_vance",
            beat_number=9,
            max_beats=10,
            emotional_state="desperate",
        )

        # Sentences should be noticeably shorter as clinical composure breaks
        assert adapted["dialect_rules"]["max_words_per_sentence"] <= 14


class TestManualOverrides:
    """Test UI slider overrides apply cleanly on top of profile."""

    def test_manual_slider_overrides(self, merc_profile):
        overrides = {
            "sentence_length": 5,      # +5 words
            "slang_density": 90,        # 90%
            "formality": 20,
        }
        adapted = adapt_profile(
            base_profile=merc_profile,
            speaker_id="merc_vance",
            listener_id="tommy_k",
            beat_number=1,
            max_beats=10,
            emotional_state="guarded",
            manual_overrides=overrides,
        )
        assert adapted["dialect_rules"]["contraction_rate"] == 0.84 or adapted["dialect_rules"]["contraction_rate"] >= 0.80
        assert adapted["dialect_rules"]["max_words_per_sentence"] > merc_profile["dialect_rules"]["max_words_per_sentence"]
