"""
Tests for LangGraph Scene State Graph, Transitions, and Fountain Formatter.
"""

from __future__ import annotations

import pytest

from writers_room.graph.scene_graph import (
    build_scene_graph,
    after_lint,
    after_critic,
    after_reject,
    after_accept,
    make_initial_state,
)
from writers_room.graph.formatter import format_screenplay, _normalise_location


class TestGraphCompilation:
    def test_build_scene_graph(self):
        graph = build_scene_graph()
        assert graph is not None

        # Check nodes are registered in the compiled graph
        nodes = graph.nodes
        expected_nodes = [
            "director_node",
            "character_node",
            "deterministic_lint_node",
            "critic_node",
            "accept_line_node",
            "reject_line_node",
            "finalize_node",
        ]
        for node_name in expected_nodes:
            assert node_name in nodes, f"Node '{node_name}' missing from graph"


class TestEdgeRouters:
    def test_after_lint_routing(self):
        # When lint passes -> goes to critic_node
        state_pass = {"lint_result": {"passed": True, "violations": []}}
        assert after_lint(state_pass) == "critic_node"

        # When lint fails -> routes to reject_line_node
        state_fail = {"lint_result": {"passed": False, "violations": ["Too long"]}}
        assert after_lint(state_fail) == "reject_line_node"

    def test_after_critic_routing(self):
        # Critic pass -> accept_line_node
        state_pass = {"critic_verdict": {"passed": True}}
        assert after_critic(state_pass) == "accept_line_node"

        # Critic fail -> reject_line_node
        state_fail = {"critic_verdict": {"passed": False}}
        assert after_critic(state_fail) == "reject_line_node"

    def test_after_reject_routing(self):
        # Retries remaining -> back to character_node
        state_retry = {"retry_count": 1, "max_retries": 2}
        assert after_reject(state_retry) == "character_node"

        # Retries exhausted -> accept_line_node (forced accept with warning)
        state_exhausted = {"retry_count": 2, "max_retries": 2}
        assert after_reject(state_exhausted) == "accept_line_node"

    def test_after_accept_routing(self):
        # More beats needed -> back to director_node
        state_continue = {"dialogue": [{"id": 1}], "max_beats": 4}
        assert after_accept(state_continue) == "director_node"

        # Max beats reached -> finalize_node
        state_finish = {"dialogue": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}], "max_beats": 4}
        assert after_accept(state_finish) == "finalize_node"


class TestFountainFormatter:
    def test_normalise_location(self):
        assert _normalise_location("Docks") == "INT. DOCKS - DAY"
        assert _normalise_location("EXT. PIER 9 - NIGHT") == "EXT. PIER 9 - NIGHT"
        assert _normalise_location("INT./EXT. CARGO HOLD - DAWN") == "INT./EXT. CARGO HOLD - DAWN"

    def test_format_screenplay_complete(self):
        mock_scene = {
            "location": "Abandoned Warehouse",
            "premise": "Two rivals meet under cover of darkness.",
            "character_profiles": {
                "merc_vance": {"display_name": "Merc Vance"},
                "dr_lena_soros": {"display_name": "Dr. Lena Soros"},
            },
            "dialogue": [
                {
                    "character_id": "merc_vance",
                    "spoken_text": "Saw three cutters past the breakwater. Ain't none of 'em flying colors.",
                    "stage_direction": "leaning against the rusted door",
                },
                {
                    "character_id": "dr_lena_soros",
                    "spoken_text": "The vessels were accounted for at 14:00 hours.",
                    "stage_direction": "glancing up from the terminal",
                },
            ],
        }

        fountain = format_screenplay(mock_scene)

        # Fountain scene heading
        assert "INT. ABANDONED WAREHOUSE - DAY" in fountain
        # Premise as action line
        assert "Two rivals meet under cover of darkness." in fountain
        # Character headers in uppercase
        assert "MERC VANCE" in fountain
        assert "DR. LENA SOROS" in fountain
        # Stage directions in parentheses
        assert "(leaning against the rusted door)" in fountain
        assert "(glancing up from the terminal)" in fountain
        # Dialogue text
        assert "Saw three cutters past the breakwater." in fountain
        assert "The vessels were accounted for at 14:00 hours." in fountain
