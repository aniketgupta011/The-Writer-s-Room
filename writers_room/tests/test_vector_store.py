"""
Tests for Memory Subsystem: ChromaDB Golden Line Vector Store.
"""

from __future__ import annotations

from pathlib import Path
import pytest

from writers_room.memory.vector_store import (
    initialize_store,
    query_golden_lines,
    _get_collection,
)

GOLDEN_LINES_DIR = Path(__file__).resolve().parent.parent / "data" / "golden_lines"


class TestVectorStore:
    def test_initialize_and_count(self):
        count = initialize_store(str(GOLDEN_LINES_DIR))
        assert count == 90

    def test_query_merc_guarded(self):
        lines = query_golden_lines("merc_vance", "guarded", n_results=3)
        assert len(lines) == 3
        # Merc lines should contain maritime/terse dialect markers
        combined = " ".join(lines)
        assert any(term in combined for term in ["cutters", "tide", "partner", "cargo", "'em", "ain't"])

    def test_query_soros_analytical(self):
        lines = query_golden_lines("dr_lena_soros", "analytical", n_results=3)
        assert len(lines) == 3
        combined = " ".join(lines)
        # Soros lines should feature clinical, scientific terminology
        assert any(term in combined.lower() for term in ["data", "subject", "protocol", "catalogue", "vector", "correlate"])

    def test_query_tommy_sarcastic(self):
        lines = query_golden_lines("tommy_k", "sarcastic", n_results=3)
        assert len(lines) == 3
        combined = " ".join(lines)
        assert any(term in combined.lower() for term in ["yo", "firewall", "crypto", "bruh", "deadass", "ramen"])
