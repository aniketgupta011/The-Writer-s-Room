"""
Tests for the deterministic voice linter (Stage A).

Uses the merc_vance profile as specified in the project plan.
"""

import sys
import os
import pytest

# Ensure the project root is on the path so we can import writers_room.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
)

from writers_room.linter.voice_linter import lint_dialogue, LintResult


# ── Sample Profile ─────────────────────────────────────────────────────────

MERC_PROFILE = {
    "character_id": "merc_vance",
    "dialect_rules": {
        "contraction_rate": "high",
        "max_words_per_sentence": 14,
        "phonetic_markers": ["'em", "ain't", "oughta"],
        "forbidden_lexicon": [
            "furthermore", "comprehend", "traumatized", "okay", "basically"
        ],
        "syntax_quirks": ["Drops subject pronouns", "Uses maritime metaphors"],
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _has_violation_containing(result: LintResult, fragment: str) -> bool:
    """Return True if any violation message contains *fragment*."""
    return any(fragment.lower() in v.lower() for v in result.violations)


# ── Tests ──────────────────────────────────────────────────────────────────

class TestPassingLine:
    """A line that passes all checks for merc_vance."""

    def test_passes_all_checks(self):
        line = "Don't trust 'em. Ain't nobody comin' to save us."
        result = lint_dialogue(line, MERC_PROFILE)
        assert result.passed is True
        assert result.violations == []
        assert result.stage == "deterministic"


class TestSentenceLength:
    """Check 1: sentence length violation."""

    def test_long_sentence_flagged(self):
        # 16 words in a single sentence — exceeds max of 14.
        line = (
            "I ain't ever seen a dockside deal go south that fast "
            "in all my born days."
        )
        result = lint_dialogue(line, MERC_PROFILE)
        assert result.passed is False
        assert _has_violation_containing(result, "Sentence length")

    def test_short_sentences_pass(self):
        line = "Get moving. Don't look back."
        result = lint_dialogue(line, MERC_PROFILE)
        # Should not fail on sentence length.
        assert not _has_violation_containing(result, "Sentence length")


class TestForbiddenLexicon:
    """Check 2: forbidden words."""

    def test_forbidden_word_detected(self):
        line = "I don't comprehend why they'd do that."
        result = lint_dialogue(line, MERC_PROFILE)
        assert result.passed is False
        assert _has_violation_containing(result, "Forbidden word")
        assert _has_violation_containing(result, "comprehend")

    def test_forbidden_word_case_insensitive(self):
        line = "FURTHERMORE, ain't nobody asked."
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "furthermore")

    def test_substring_not_flagged(self):
        """'comprehension' should NOT trigger 'comprehend'."""
        line = "Ain't got no comprehension of that."
        result = lint_dialogue(line, MERC_PROFILE)
        # 'comprehension' ≠ 'comprehend' due to word boundaries.
        assert not _has_violation_containing(result, "comprehend")

    def test_okay_flagged(self):
        line = "That ain't okay with me."
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "okay")


class TestContractionRate:
    """Check 3: contraction rate."""

    def test_too_formal_for_high_profile(self):
        """A line with no contractions should fail for 'high' contraction rate."""
        line = (
            "The ship sailed past the harbor without stopping. "
            "The crew did not respond to signals."
        )
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "Contraction rate too low")

    def test_high_contraction_passes(self):
        """A line heavy with contractions should pass for 'high' profile."""
        line = "Don't know why they'd bother. Can't trust 'em."
        result = lint_dialogue(line, MERC_PROFILE)
        assert not _has_violation_containing(result, "Contraction rate")

    def test_low_profile_flags_too_many_contractions(self):
        low_profile = {
            "character_id": "dr_soros",
            "dialect_rules": {
                "contraction_rate": "low",
                "max_words_per_sentence": 25,
                "phonetic_markers": [],
                "forbidden_lexicon": [],
            },
        }
        # Heavy contractions (> 0.20 rate to trigger "low" violation)
        line = "I can't believe she'd do that. They won't listen and I don't care anyway."
        result = lint_dialogue(line, low_profile)
        assert _has_violation_containing(result, "Contraction rate too high")

    def test_medium_profile_acceptable_range(self):
        med_profile = {
            "character_id": "tommy_k",
            "dialect_rules": {
                "contraction_rate": "medium",
                "max_words_per_sentence": 10,
                "phonetic_markers": ["bruh"],
                "forbidden_lexicon": [],
            },
        }
        # One contraction in ~8 words → ~0.125 rate (within 0.05–0.15)
        line = "Can't stop now bruh. Keep going."
        result = lint_dialogue(line, med_profile)
        assert not _has_violation_containing(result, "Contraction rate")


class TestPhoneticMarkers:
    """Check 4: phonetic marker presence."""

    def test_missing_marker_warning(self):
        """A long line with no phonetic markers triggers a soft warning."""
        # >8 words, no markers
        line = "The cargo ship dropped anchor near the old wharf at dawn today."
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "Phonetic marker warning")

    def test_marker_present_no_warning(self):
        line = "Told 'em to stay clear of the break. Ain't safe there."
        result = lint_dialogue(line, MERC_PROFILE)
        assert not _has_violation_containing(result, "Phonetic marker")

    def test_short_line_skips_marker_check(self):
        # ≤8 words — marker check should be skipped.
        line = "Get out of my way."
        result = lint_dialogue(line, MERC_PROFILE)
        assert not _has_violation_containing(result, "Phonetic marker")


class TestAICadence:
    """Check 5: AI cadence detection."""

    def test_i_understand_at_start(self):
        line = "I understand your frustration, but we must proceed."
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "I understand")

    def test_its_important_to(self):
        line = "It's important to remember we ain't alone out here."
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "important to")

    def test_its_worth_noting(self):
        line = "It's worth noting that the tides shifted early."
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "worth noting")

    def test_semicolon_balanced_clause(self):
        line = "The dock was empty; however, the lights were on."
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "semicolon")

    def test_as_a_role_pattern(self):
        line = "As a captain, I'd know if something were off."
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "As a")

    def test_right_softener(self):
        line = "We're heading to the dock at midnight, right?"
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "right?")

    def test_llm_triad(self):
        line = "Grab the rope, the anchor, and the map."
        result = lint_dialogue(line, MERC_PROFILE)
        assert _has_violation_containing(result, "triad")


class TestEdgeCases:
    """Edge cases: empty string, single word, all caps."""

    def test_empty_string(self):
        result = lint_dialogue("", MERC_PROFILE)
        assert result.passed is True
        assert result.violations == []

    def test_whitespace_only(self):
        result = lint_dialogue("   ", MERC_PROFILE)
        assert result.passed is True

    def test_single_word(self):
        result = lint_dialogue("Run!", MERC_PROFILE)
        assert result.passed is True

    def test_single_forbidden_word(self):
        result = lint_dialogue("Okay.", MERC_PROFILE)
        assert result.passed is False
        assert _has_violation_containing(result, "okay")

    def test_all_caps(self):
        """ALL CAPS should still work (case-insensitive checks)."""
        result = lint_dialogue("BASICALLY WE'RE DONE.", MERC_PROFILE)
        assert result.passed is False
        assert _has_violation_containing(result, "basically")

    def test_stage_always_deterministic(self):
        result = lint_dialogue("Anything.", MERC_PROFILE)
        assert result.stage == "deterministic"
