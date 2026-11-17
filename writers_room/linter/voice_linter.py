"""
Deterministic Voice Linter — Stage A of the two-stage dialogue critic.

Takes a dialogue line and a character voice profile, runs five deterministic
checks, and returns a LintResult.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Result types (standalone; if data.schemas is available import from there)
# ---------------------------------------------------------------------------

@dataclass
class LintResult:
    """Outcome of the deterministic linting pass."""
    passed: bool
    violations: list[str] = field(default_factory=list)
    stage: str = "deterministic"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common English contractions — apostrophe-containing words that count.
_CONTRACTION_RE = re.compile(
    r"""(?ix)               # ignore case, verbose
    \b
    (?:
        \w+'t             # don't, won't, can't, isn't, aren't, wasn't, weren't, hasn't, haven't, hadn't, couldn't, shouldn't, wouldn't, mustn't, needn't, oughtn't, didn't
      | \w+'re            # you're, we're, they're
      | \w+'ve            # I've, you've, we've, they've, could've, should've, would've
      | \w+'ll            # I'll, you'll, he'll, she'll, we'll, they'll, it'll
      | \w+'d             # I'd, you'd, he'd, she'd, we'd, they'd, it'd
      | I'm               # I'm
      | \w+'m             # (catch-all for I'm variants)
      | \w+'s             # he's, she's, it's, that's, what's, there's, here's, who's, let's
      | ain't
      | ma'am
      | y'all
      | 'em               # 'em (them)
      | 'cause            # 'cause (because)
      | 'nuff             # 'nuff (enough)
      | 'bout             # 'bout (about)
      | 'til              # 'til (until)
      | o'clock
    )
    \b
    """,
)

# Sentence-splitting regex — split on .  !  ?  …  (handling ellipsis as one)
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?…])\s+|(?<=\.\.\.)\s+')

# AI cadence patterns
_AI_CADENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)^I understand\b"),
     "AI cadence: starts with 'I understand'"),
    (re.compile(r"(?i)\bIt'?s important to\b"),
     "AI cadence: contains 'It's important to'"),
    (re.compile(r"(?i)\bIt'?s worth noting\b"),
     "AI cadence: contains 'It's worth noting'"),
    (re.compile(r";.+(?:however|but|yet|nonetheless|nevertheless),"),
     "AI cadence: balanced semicolon-separated clauses"),
    (re.compile(r"(?i)\bAs an?\s+\w+\b"),
     "AI cadence: 'As a [role]' pattern"),
    (re.compile(r"(?i),?\s*right\?\s*$"),
     "AI cadence: ends with 'right?' softener"),
]

# LLM triad: "X, Y, and Z" — three comma-separated items ending with "and Z"
_LLM_TRIAD_RE = re.compile(
    r"(?i)\b\w+(?:\s+\w+)*,\s+\w+(?:\s+\w+)*,\s+and\s+\w+(?:\s+\w+)*\b"
)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, handling common dialogue punctuation."""
    # Normalise ellipsis character to three dots for uniform splitting.
    text = text.replace("…", "...")

    # Split on sentence-ending punctuation followed by whitespace.
    parts = re.split(r'(?<=[.!?])\s+|(?<=\.\.\.)\s+', text)

    # Strip and remove blanks.
    return [s.strip() for s in parts if s.strip()]


def _word_count(text: str) -> int:
    """Return the number of words in *text*."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_sentence_length(text: str, max_words: int) -> list[str]:
    """Check 1: every sentence must be ≤ *max_words* words."""
    violations: list[str] = []
    for sentence in _split_sentences(text):
        wc = _word_count(sentence)
        if wc > max_words:
            violations.append(
                f"Sentence length: {wc} words (max {max_words}): "
                f"\"{sentence}\""
            )
    return violations


def _check_forbidden_lexicon(text: str, forbidden: list[str]) -> list[str]:
    """Check 2: no word from *forbidden* must appear (word-boundary match)."""
    violations: list[str] = []
    for word in forbidden:
        pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            violations.append(
                f"Forbidden word: \"{m.group()}\" at position {m.start()}"
            )
    return violations


def _check_contraction_rate(text: str, expected_rate) -> list[str]:
    """Check 3: contraction density must match the profile's expected rate.

    *expected_rate* may be a string ("high"/"medium"/"low") or a numeric float
    (0.0–1.0).  Numeric values are mapped: ≥ 0.5 → "high", 0.15–0.5 → "medium",
    < 0.15 → "low".
    """
    words = text.split()
    total = len(words)
    # Skip contraction-rate checks for very short lines (≤ 5 words) — too
    # few words to produce a meaningful ratio.
    if total <= 5:
        return []

    contractions = _CONTRACTION_RE.findall(text)
    rate = len(contractions) / total

    # Normalise numeric contraction_rate to a label
    if isinstance(expected_rate, (int, float)):
        if expected_rate >= 0.5:
            expected_rate = "high"
        elif expected_rate >= 0.15:
            expected_rate = "medium"
        else:
            expected_rate = "low"

    violations: list[str] = []
    if expected_rate == "high" and rate < 0.05:
        violations.append(
            f"Contraction rate too low: {rate:.2f} (expected ≥ 0.05 for 'high' profile)"
        )
    elif expected_rate == "low" and rate > 0.20:
        violations.append(
            f"Contraction rate too high: {rate:.2f} (expected ≤ 0.20 for 'low' profile)"
        )
    elif expected_rate == "medium":
        if rate < 0.03:
            violations.append(
                f"Contraction rate too low: {rate:.2f} (expected 0.03–0.20 for 'medium' profile)"
            )
        elif rate > 0.20:
            violations.append(
                f"Contraction rate too high: {rate:.2f} (expected 0.03–0.20 for 'medium' profile)"
            )
    return violations


def _check_phonetic_markers(text: str, markers: list[str]) -> list[str]:
    """Check 4: at least one phonetic marker should appear in longer lines."""
    if _word_count(text) <= 8:
        return []  # Only check lines longer than 8 words.
    if not markers:
        return []

    text_lower = text.lower()
    for marker in markers:
        if marker.lower() in text_lower:
            return []

    return [
        f"Phonetic marker warning: none of {markers} found in line "
        f"({_word_count(text)} words). This is a soft warning."
    ]


def _check_ai_cadence(text: str) -> list[str]:
    """Check 5: detect common LLM-generated cadence tells."""
    violations: list[str] = []
    for pattern, message in _AI_CADENCE_PATTERNS:
        if pattern.search(text):
            violations.append(message)

    if _LLM_TRIAD_RE.search(text):
        violations.append("AI cadence: LLM triad pattern (X, Y, and Z)")

    return violations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lint_dialogue(text: str, profile: Dict[str, Any]) -> LintResult:
    """Run all deterministic linting checks on *text* for the given *profile*.

    Parameters
    ----------
    text : str
        A single dialogue line to validate.
    profile : dict
        A character voice profile. Expected shape::

            {
                "character_id": "merc_vance",
                "dialect_rules": {
                    "contraction_rate": "high",
                    "max_words_per_sentence": 14,
                    "phonetic_markers": ["'em", "ain't", "oughta"],
                    "forbidden_lexicon": ["furthermore", "comprehend", ...],
                    "syntax_quirks": [...]
                }
            }

    Returns
    -------
    LintResult
        With ``passed=True`` when *all* checks pass (violations list empty).
    """
    # Handle edge cases
    if not text or not text.strip():
        return LintResult(passed=True, violations=[], stage="deterministic")

    rules = profile.get("dialect_rules", {})
    violations: list[str] = []

    # Check 1 — Sentence length
    max_words = rules.get("max_words_per_sentence")
    if max_words is not None:
        violations.extend(_check_sentence_length(text, max_words))

    # Check 2 — Forbidden lexicon
    forbidden = rules.get("forbidden_lexicon", [])
    if forbidden:
        violations.extend(_check_forbidden_lexicon(text, forbidden))

    # Check 3 — Contraction rate
    contraction_rate = rules.get("contraction_rate")
    if contraction_rate:
        violations.extend(_check_contraction_rate(text, contraction_rate))

    # Check 4 — Phonetic marker presence
    markers = rules.get("phonetic_markers", [])
    violations.extend(_check_phonetic_markers(text, markers))

    # Check 5 — AI cadence detection
    violations.extend(_check_ai_cadence(text))

    return LintResult(
        passed=len(violations) == 0,
        violations=violations,
        stage="deterministic",
    )
