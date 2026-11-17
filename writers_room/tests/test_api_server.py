"""
Tests for FastAPI REST and WebSocket API Endpoints.
"""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from writers_room.api.server import app, _scenes


@pytest.fixture
def client():
    return TestClient(app)


class TestAPIRoutes:
    def test_list_characters(self, client):
        response = client.get("/api/characters")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 3
        cids = [c["character_id"] for c in data]
        assert "merc_vance" in cids
        assert "dr_lena_soros" in cids
        assert "tommy_k" in cids

        merc = next(c for c in data if c["character_id"] == "merc_vance")
        assert merc["max_words_per_sentence"] == 14
        assert len(merc["phonetic_markers"]) > 0

    def test_ui_and_static_serving(self, client):
        response = client.get("/ui")
        assert response.status_code == 200
        assert "<title>The Writers' Room" in response.text

        css_resp = client.get("/ui/static/styles.css")
        assert css_resp.status_code == 200
        assert "screenplay-pane" in css_resp.text

        js_resp = client.get("/ui/static/app.js")
        assert js_resp.status_code == 200
        assert "startScene" in js_resp.text

    def test_dialect_update(self, client):
        # Create a mock scene in memory
        scene_id = "test_scene_123"
        _scenes[scene_id] = {
            "scene_id": scene_id,
            "character_profiles": {
                "merc_vance": {
                    "dialect_rules": {
                        "max_words_per_sentence": 14,
                        "contraction_rate": 0.85,
                    }
                }
            },
            "beats": [],
            "dialogue": [],
            "errors": [],
            "is_complete": False,
            "screenplay": "",
        }

        # Update sentence length
        resp = client.put(
            f"/api/scene/{scene_id}/dialect",
            json={"character_id": "merc_vance", "modifier": "sentence_length", "value": -3},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

        # Check updated max words in profile
        updated_max = _scenes[scene_id]["character_profiles"]["merc_vance"]["dialect_rules"]["max_words_per_sentence"]
        assert updated_max == 11  # 14 - 3 = 11

    def test_get_screenplay_endpoint(self, client):
        scene_id = "test_screenplay_scene"
        _scenes[scene_id] = {
            "scene_id": scene_id,
            "location": "Warehouse Dock",
            "premise": "A tense negotiation.",
            "character_profiles": {"merc_vance": {"display_name": "Merc Vance"}},
            "dialogue": [
                {
                    "character_id": "merc_vance",
                    "spoken_text": "Ain't got time for this.",
                    "stage_direction": "spits into the dark water",
                }
            ],
            "screenplay": "",
            "is_complete": True,
        }

        resp = client.get(f"/api/scene/{scene_id}/screenplay")
        assert resp.status_code == 200
        text = resp.json()["screenplay"]
        assert "INT. WAREHOUSE DOCK - DAY" in text
        assert "MERC VANCE" in text
        assert "Ain't got time for this." in text
