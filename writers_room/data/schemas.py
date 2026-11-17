"""Pydantic v2 models for the Writers' Room dialogue system."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Voice-profile schemas ────────────────────────────────────────────


class DialectRules(BaseModel):
    """Low-level linguistic knobs that shape how a character speaks."""

    contraction_rate: float | str = Field(
        ...,
        description="Numeric 0.0–1.0, or label 'high'/'medium'/'low'",
    )
    max_words_per_sentence: int = Field(
        ..., gt=0, description="Hard cap on sentence length"
    )
    syntax_quirks: list[str] = Field(
        default_factory=list,
        description="Recurring syntactic patterns, e.g. 'drops subject pronouns'",
    )
    phonetic_markers: list[str] = Field(
        default_factory=list,
        description="Colloquial spellings the character uses, e.g. 'em, ain't",
    )
    forbidden_lexicon: list[str] = Field(
        default_factory=list,
        description="Words this character would never use",
    )


class SubtextBehavior(BaseModel):
    """How the character's speech mutates under specific emotional states."""

    when_lying: str = Field(..., description="Speech pattern shift when lying")
    when_angry: str = Field(..., description="Speech pattern shift when angry")
    when_desperate: str = Field(
        ..., description="Speech pattern shift when desperate"
    )
    when_guarded: str = Field(
        ..., description="Speech pattern shift when guarded"
    )
    when_mocking: str = Field(
        ..., description="Speech pattern shift when mocking"
    )


class CharacterVoiceProfile(BaseModel):
    """Complete voice profile for a single character."""

    character_id: str = Field(..., description="Unique slug, e.g. 'merc_vance'")
    display_name: str = Field(..., description="Human-readable name")
    archetype: str = Field(
        ..., description="One-line character concept, e.g. 'Cynical Dockside Smuggler'"
    )
    dialect_rules: DialectRules
    subtext_behavior: SubtextBehavior
    backstory: str = Field(
        ...,
        description="2-3 sentence backstory for the character",
    )
    emotional_state_default: str = Field(
        ...,
        description="Baseline emotional state, e.g. 'guarded'",
    )


# ── Golden-line schemas ──────────────────────────────────────────────


class GoldenLine(BaseModel):
    """A single exemplar line that captures a character's voice."""

    line_id: str = Field(..., description="Unique id, e.g. 'mv_001'")
    text: str = Field(..., description="The spoken line")
    emotion: str = Field(
        ..., description="Dominant emotion: angry, guarded, sarcastic, etc."
    )
    context: str = Field(
        ..., description="Brief situational context for the line"
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Thematic / stylistic tags",
    )


class GoldenLineCollection(BaseModel):
    """All golden lines for one character."""

    character_id: str
    lines: list[GoldenLine]


# ── Scene / pipeline schemas ─────────────────────────────────────────


class SceneBeat(BaseModel):
    """One dramatic beat inside a scene."""

    beat_number: int = Field(..., ge=1)
    next_speaker: str = Field(
        ..., description="character_id of who speaks next"
    )
    tactical_intent: str = Field(
        ...,
        description="What the speaker is trying to achieve with this beat",
    )
    emotional_state: str = Field(
        ..., description="Speaker's emotional state during this beat"
    )
    physical_action: str = Field(
        "", description="Optional stage direction / physical action"
    )


class DialogueLine(BaseModel):
    """A single line of dialogue produced by the pipeline."""

    character_id: str
    spoken_text: str
    stage_direction: str = ""
    beat_number: int = Field(..., ge=1)
    timestamp: str = Field(
        "",
        description="ISO-8601 timestamp of when the line was generated",
    )


class LintResult(BaseModel):
    """Output of a lint / validation pass."""

    passed: bool
    violations: list[str] = Field(default_factory=list)
    stage: str = Field(
        ..., description="Pipeline stage that produced this result"
    )


class CriticVerdict(BaseModel):
    """Output of the critic / similarity check."""

    passed: bool
    feedback: str
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0.0 = no resemblance, 1.0 = identical to golden lines",
    )


class SceneState(BaseModel):
    """Full mutable state of a scene in progress."""

    scene_id: str
    premise: str
    location: str
    beats: list[SceneBeat] = Field(default_factory=list)
    dialogue: list[DialogueLine] = Field(default_factory=list)
    active_characters: list[str] = Field(
        default_factory=list,
        description="character_ids of characters present in the scene",
    )
