"""
Central configuration for the Writers' Room system.
"""

import os
from pathlib import Path

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
    load_dotenv()
except ImportError:
    pass

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
VOICE_PROFILES_DIR = DATA_DIR / "voice_profiles"
GOLDEN_LINES_DIR = DATA_DIR / "golden_lines"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

# ── Model Configuration ───────────────────────────────────────────────────
DIRECTOR_MODEL = os.getenv("DIRECTOR_MODEL", "gemini-3.5-flash-lite")
CHARACTER_MODEL = os.getenv("CHARACTER_MODEL", "gemini-3.5-flash-lite")
CRITIC_MODEL = os.getenv("CRITIC_MODEL", "gemini-3.5-flash-lite")
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.8-flash"]

# ── Pipeline Defaults ─────────────────────────────────────────────────────
DEFAULT_MAX_BEATS = 10
DEFAULT_MAX_RETRIES = 2
GOLDEN_LINES_PER_QUERY = 3

# ── API Configuration ─────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
CORS_ORIGINS = ["*"]

# ── Voice Linter Thresholds ───────────────────────────────────────────────
CONTRACTION_THRESHOLDS = {
    "high": {"min": 0.05, "max": 1.0},
    "medium": {"min": 0.03, "max": 0.15},
    "low": {"min": 0.0, "max": 0.10},
}

CRITIC_PASS_THRESHOLD = 0.7

# ── AI Cadence Patterns (things LLMs say that real characters don't) ─────
AI_CADENCE_PATTERNS = [
    r"^I understand\b",
    r"\bIt'?s important to\b",
    r"\bIt'?s worth noting\b",
    r"\bAs a [a-z]+,?\s",
    r"\bIn (this|that) context\b",
    r"\bLet me be clear\b",
    r"\bI appreciate (your|the)\b",
    r"\bAt the end of the day\b",
    r"\bright\?$",
]

