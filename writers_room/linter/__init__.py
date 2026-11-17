"""
writers_room.linter — Two-stage dialogue linting and critique.

Stage A: Deterministic voice linter (lint_dialogue)
Stage B: LLM style critic (critique_dialogue)
"""

from .voice_linter import lint_dialogue, LintResult
from .critic import critique_dialogue, CriticVerdict

__all__ = [
    "lint_dialogue",
    "LintResult",
    "critique_dialogue",
    "CriticVerdict",
]
