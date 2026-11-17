"""
FastAPI backend for the Writers' Room — Multi-Agent Dialect & Voice Orchestrator.

Endpoints
---------
REST:
    POST /api/scene/start          — Start a new scene
    GET  /api/scene/{id}/status    — Get scene progress
    GET  /api/scene/{id}/screenplay — Get formatted screenplay
    GET  /api/characters           — List available characters
    PUT  /api/scene/{id}/dialect   — Adjust dialect sliders

WebSocket:
    WS   /ws/scene/{id}            — Real-time event stream
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── App setup ────────────────────────────────────────────────────────

logger = logging.getLogger("writers_room.api")

app = FastAPI(
    title="The Writers' Room",
    description="Multi-Agent Dialect & Voice Orchestrator",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Path constants ───────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UI_DIR = _PROJECT_ROOT / "ui"
_VOICE_PROFILES_DIR = _PROJECT_ROOT / "data" / "voice_profiles"
_GOLDEN_LINES_DIR = _PROJECT_ROOT / "data" / "golden_lines"

# ── In-memory scene store ────────────────────────────────────────────

_scenes: dict[str, dict[str, Any]] = {}
_scene_queues: dict[str, asyncio.Queue] = {}
_scene_tasks: dict[str, asyncio.Task] = {}

# ── Request / response models ────────────────────────────────────────


class SceneStartRequest(BaseModel):
    premise: str
    location: str
    characters: list[str]
    max_beats: int = Field(default=10, ge=2, le=30)


class SceneStartResponse(BaseModel):
    scene_id: str
    status: str
    characters: list[str]
    max_beats: int


class SceneStatusResponse(BaseModel):
    scene_id: str
    is_complete: bool
    beat_count: int
    dialogue_count: int
    max_beats: int
    errors: list[str]


class DialectUpdateRequest(BaseModel):
    character_id: str
    modifier: str          # e.g. "slang_density", "max_words_per_sentence"
    value: float | int | str


class CharacterInfo(BaseModel):
    character_id: str
    display_name: str
    archetype: str
    max_words_per_sentence: int
    contraction_rate: float | str
    forbidden_count: int
    phonetic_markers: list[str]


# ── Helpers ──────────────────────────────────────────────────────────


def _load_profiles(character_ids: list[str]) -> dict[str, dict]:
    """Load voice profiles from disk for the requested character IDs."""
    profiles: dict[str, dict] = {}
    for cid in character_ids:
        path = _VOICE_PROFILES_DIR / f"{cid}.json"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Voice profile not found: {cid}",
            )
        with open(path, "r", encoding="utf-8") as f:
            profiles[cid] = json.load(f)
    return profiles


def _all_profiles() -> list[dict]:
    """Load every voice profile from disk."""
    profiles = []
    for path in sorted(_VOICE_PROFILES_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            profiles.append(json.load(f))
    return profiles


async def _run_scene(scene_id: str) -> None:
    """Background task: run the LangGraph pipeline for a scene."""
    scene = _scenes[scene_id]
    queue = _scene_queues[scene_id]

    try:
        from writers_room.memory.vector_store import initialize_store
        from writers_room.memory import vector_store as vs_module
        from writers_room.agents.director import DirectorAgent
        from writers_room.agents.character import CharacterAgent
        from writers_room.graph.scene_graph import build_scene_graph

        # Ensure vector store is initialized
        initialize_store(str(_GOLDEN_LINES_DIR))

        # Create agents
        director = DirectorAgent()
        char_pool: dict[str, CharacterAgent] = {}
        for cid, profile in scene["character_profiles"].items():
            char_pool[cid] = CharacterAgent(
                character_id=cid,
                voice_profile=profile,
                vector_store=vs_module,
            )

        # Build and invoke the graph
        graph = build_scene_graph()

        initial_state = {
            "scene_id": scene_id,
            "premise": scene["premise"],
            "location": scene["location"],
            "active_characters": scene["active_characters"],
            "character_profiles": scene["character_profiles"],
            "beats": [],
            "dialogue": [],
            "current_beat": None,
            "current_line": None,
            "current_adapted_profile": None,
            "lint_result": None,
            "critic_verdict": None,
            "retry_count": 0,
            "max_retries": 2,
            "max_beats": scene["max_beats"],
            "errors": [],
            "is_complete": False,
            "_director": director,
            "_character_pool": char_pool,
            "_event_queue": queue,
        }

        final_state = await graph.ainvoke(initial_state)

        # Update the scene store with results
        scene["beats"] = final_state.get("beats", [])
        scene["dialogue"] = final_state.get("dialogue", [])
        scene["errors"] = final_state.get("errors", [])
        scene["is_complete"] = True
        scene["screenplay"] = final_state.get("screenplay", "")

        # Signal completion
        await queue.put({"type": "complete", "data": {"screenplay": scene["screenplay"]}})

    except Exception as exc:
        logger.exception("Scene %s failed", scene_id)
        scene["errors"].append(f"Pipeline error: {exc}")
        scene["is_complete"] = True
        await queue.put({"type": "error", "data": {"error": str(exc)}})


# ── REST endpoints ───────────────────────────────────────────────────


@app.post("/api/scene/start", response_model=SceneStartResponse)
async def start_scene(req: SceneStartRequest):
    """Create a scene and begin graph execution in background."""
    profiles = _load_profiles(req.characters)
    scene_id = str(uuid.uuid4())[:8]

    scene = {
        "scene_id": scene_id,
        "premise": req.premise,
        "location": req.location,
        "active_characters": req.characters,
        "character_profiles": profiles,
        "max_beats": req.max_beats,
        "beats": [],
        "dialogue": [],
        "errors": [],
        "is_complete": False,
        "screenplay": "",
    }
    _scenes[scene_id] = scene
    _scene_queues[scene_id] = asyncio.Queue()

    # Launch the pipeline as a background task
    task = asyncio.create_task(_run_scene(scene_id))
    _scene_tasks[scene_id] = task

    return SceneStartResponse(
        scene_id=scene_id,
        status="running",
        characters=req.characters,
        max_beats=req.max_beats,
    )


@app.get("/api/scene/{scene_id}/status", response_model=SceneStatusResponse)
async def scene_status(scene_id: str):
    """Check current scene progress."""
    if scene_id not in _scenes:
        raise HTTPException(404, detail=f"Scene {scene_id} not found")
    scene = _scenes[scene_id]
    return SceneStatusResponse(
        scene_id=scene_id,
        is_complete=scene["is_complete"],
        beat_count=len(scene["beats"]),
        dialogue_count=len(scene["dialogue"]),
        max_beats=scene["max_beats"],
        errors=scene["errors"][-10:],  # last 10 errors
    )


@app.get("/api/scene/{scene_id}/screenplay")
async def get_screenplay(scene_id: str):
    """Return the formatted Fountain screenplay."""
    if scene_id not in _scenes:
        raise HTTPException(404, detail=f"Scene {scene_id} not found")
    scene = _scenes[scene_id]

    if scene["screenplay"]:
        return {"screenplay": scene["screenplay"]}

    # Generate on-the-fly if not cached
    from writers_room.graph.formatter import format_screenplay
    text = format_screenplay(scene)
    return {"screenplay": text}


@app.get("/api/characters", response_model=list[CharacterInfo])
async def list_characters():
    """List all available character profiles."""
    result = []
    for profile in _all_profiles():
        dialect = profile.get("dialect_rules", {})
        result.append(CharacterInfo(
            character_id=profile["character_id"],
            display_name=profile.get("display_name", profile["character_id"]),
            archetype=profile.get("archetype", ""),
            max_words_per_sentence=dialect.get("max_words_per_sentence", 20),
            contraction_rate=dialect.get("contraction_rate", 0.5),
            forbidden_count=len(dialect.get("forbidden_lexicon", [])),
            phonetic_markers=dialect.get("phonetic_markers", []),
        ))
    return result


@app.put("/api/scene/{scene_id}/dialect")
async def update_dialect(scene_id: str, req: DialectUpdateRequest):
    """Dynamically adjust a character's dialect parameter mid-scene."""
    if scene_id not in _scenes:
        raise HTTPException(404, detail=f"Scene {scene_id} not found")

    scene = _scenes[scene_id]
    profiles = scene.get("character_profiles", {})

    if req.character_id not in profiles:
        raise HTTPException(404, detail=f"Character {req.character_id} not in scene")

    profile = profiles[req.character_id]
    dialect = profile.setdefault("dialect_rules", {})

    # Map slider names to profile keys
    modifier_map = {
        "slang_density": "contraction_rate",
        "formality": "contraction_rate",
        "sentence_length": "max_words_per_sentence",
        "emotional_hostility": None,  # stored as custom metadata
    }

    target_key = modifier_map.get(req.modifier, req.modifier)

    if target_key == "max_words_per_sentence":
        # Apply delta: base ± modifier
        current = dialect.get("max_words_per_sentence", 14)
        dialect["max_words_per_sentence"] = max(3, current + int(req.value))
    elif target_key == "contraction_rate":
        dialect["contraction_rate"] = float(req.value)
    elif target_key is None:
        # Store as custom metadata for prompt injection
        profile.setdefault("_dynamic_modifiers", {})[req.modifier] = req.value
    else:
        dialect[target_key] = req.value

    return {
        "status": "updated",
        "character_id": req.character_id,
        "modifier": req.modifier,
        "value": req.value,
    }


# ── WebSocket endpoint ───────────────────────────────────────────────


@app.websocket("/ws/scene/{scene_id}")
async def websocket_scene(ws: WebSocket, scene_id: str):
    """Stream real-time scene events to the frontend."""
    await ws.accept()

    if scene_id not in _scenes:
        await ws.send_json({"type": "error", "data": {"error": f"Scene {scene_id} not found"}})
        await ws.close()
        return

    queue = _scene_queues.get(scene_id)
    if queue is None:
        await ws.send_json({"type": "error", "data": {"error": "No event queue for this scene"}})
        await ws.close()
        return

    try:
        while True:
            # Wait for next event from the pipeline
            try:
                event = await asyncio.wait_for(queue.get(), timeout=120.0)
            except asyncio.TimeoutError:
                # Send a heartbeat to keep the connection alive
                await ws.send_json({"type": "heartbeat", "data": {}})
                continue

            await ws.send_json(event)

            # Stop streaming on completion or error
            if event.get("type") in ("complete", "error"):
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for scene %s", scene_id)
    except Exception as exc:
        logger.exception("WebSocket error for scene %s", scene_id)
        try:
            await ws.send_json({"type": "error", "data": {"error": str(exc)}})
        except Exception:
            pass


# ── Static file & UI serving ──────────────────────────────────────────


@app.get("/")
@app.get("/ui")
@app.get("/ui/")
@app.get("/index.html")
@app.get("/ui/index.html")
async def serve_ui():
    """Serve the main UI page on root and /ui paths."""
    index_path = _UI_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>UI not found. Build the frontend first.</h1>", status_code=404)


@app.get("/styles.css")
@app.get("/ui/styles.css")
async def serve_styles():
    """Serve styles.css directly."""
    css_path = _UI_DIR / "styles.css"
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    return HTMLResponse("", status_code=404)


@app.get("/app.js")
@app.get("/ui/app.js")
async def serve_js():
    """Serve app.js directly."""
    js_path = _UI_DIR / "app.js"
    if js_path.exists():
        return FileResponse(js_path, media_type="application/javascript")
    return HTMLResponse("", status_code=404)


# Mount static files (CSS, JS)
if _UI_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=str(_UI_DIR)), name="ui-static")
    app.mount("/static", StaticFiles(directory=str(_UI_DIR)), name="static")
