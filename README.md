# 🎬 The Writers' Room

**Multi-Agent Dialect & Voice Orchestrator** — A LangGraph-powered system that isolates character voices into separate execution contexts while maintaining unified scene direction, eliminating LLM voice homogenization.

> ⚠️ **Project Status: 50% MVP (Milestone 1–3 Complete)**  
> This repository contains the functional Phase 1 Minimum Viable Product demonstrating core multi-agent voice isolation, deterministic acoustic linting, few-shot RAG golden-line matching, dynamic dialect adaptation, and live WebSocket screenplay streaming. Full project completion, multi-modal TTS voice synthesis, neural linters, and full-episode narrative arcs are planned for Phase 2.

---

## 📊 MVP Progress & Roadmap

### ✅ Completed in 50% MVP (Phases 1 & 2)
- [x] **State Machine & Orchestration**: LangGraph cyclic scene state machine (`Director -> Character -> Linter -> Critic -> Dialogue Acceptance / Retry`).
- [x] **Context Isolation**: Characters run in isolated execution contexts with dedicated system prompts and separate token budgets—preventing voice bleeding.
- [x] **Character Profiles & Canon Golden Lines**:
  - [x] 3 contrasting character profiles (`merc_vance`, `dr_lena_soros`, `tommy_k`).
  - [x] 90 canonical golden lines (30 per character) indexed with emotional state, subtext, and context tags.
- [x] **Two-Stage Linguistic Critic**:
  - [x] **Stage A (Deterministic Python Linter)**: Enforces hard constraints on sentence length, forbidden lexicon, contraction rates, phonetic markers, and detects AI cadence tells (e.g., therapy-speak, triads, balanced semicolons).
  - [x] **Stage B (LLM Style Critic)**: Semantic comparison against 3 retrieved golden lines for cadence and register authenticity.
  - [x] **Automated Retry Loop**: Max 2 retries with targeted linter feedback injected back into the character's prompt.
- [x] **Dynamic Dialect Adaptation (Milestone 3)**:
  - [x] Stress escalation curves (compresses sentence length and dialect markers under pressure).
  - [x] Social authority matrix & code-switching (characters adjust slang density depending on whom they are addressing).
- [x] **Interactive Split-Screen UI & WebSocket Backend**:
  - [x] FastAPI server serving REST endpoints and bi-directional WebSocket streams.
  - [x] Live screenplay pane rendering Fountain-formatted dialogue, sluglines, and stage directions.
  - [x] Dynamic dialect modifier sliders updating running state in real time.
  - [x] Acoustic linter / critic rejection log detailing rejected lines and reasons.
- [x] **Resilient Gemini API Integration**:
  - [x] Configured for current production Gemini models (`gemini-3.5-flash-lite`) with automatic fallback to `gemini-3.1-flash-lite` and `gemini-3.8-flash`.
- [x] **Automated Test Suite**: 58 automated unit and integration tests passing across linter, dialect adapter, vector store, state graph, and voice separation benchmarks.

---

### 🔮 Planned for Final Completion (Phase 3 & 4)
- [ ] **Multi-Modal Audio & Voice Synthesis**:
  - [ ] ElevenLabs / Google Cloud TTS integration with custom voice clones matching dialect rhythm and emotional states.
  - [ ] Phonetic accent mapping (preserving spoken slang and eye-dialects during audio synthesis).
- [ ] **Multi-Scene & Full Episode Arcs**:
  - [ ] Multi-act narrative scene graphs with long-term character relationship memory.
  - [ ] Subtext & secret knowledge tracking (agents knowing secrets their scene partners don't).
  - [ ] Multi-character ensemble scenes (support for 5+ characters with dynamic turn-taking and interruptions).
- [ ] **Interactive "Human-in-the-Loop" Director Mode**:
  - [ ] Live beat intervention: Pause generation to modify director intentions or manually edit lines before graph resumption.
  - [ ] Branching dialogue tree explorer with rollback and alternate timeline generation.
- [ ] **Neural Linter & Metric Visualizations**:
  - [ ] Fine-tuned lightweight DeBERTa/RoBERTa classifier for dialect fidelity scoring alongside regex.
  - [ ] Real-time dialect deviation radar: live radial charts tracking linguistic drift, sentiment, and lexical diversity.
  - [ ] In-browser Custom Character Builder: Upload a script/novel PDF to automatically extract voice profiles and golden lines via RAG.
- [ ] **Screenplay Export & Production Tooling**:
  - [ ] Native Final Draft (`.fdx`) and standard Hollywood PDF screenplay export.
  - [ ] PostgreSQL / SQLite persistent database for screenplay versioning and user project workspaces.
  - [ ] Docker Compose package with bundled ChromaDB persistence.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Scene State Graph                      │
│                    (LangGraph Python)                     │
│                                                          │
│   ┌─────────┐    ┌───────────┐    ┌──────────────────┐  │
│   │ Director │───▸│ Character │───▸│ Deterministic    │  │
│   │  Agent   │    │   Agent   │    │   Voice Linter   │  │
│   │ (Gemini  │    │  (Gemini  │    │  (Python/Regex)  │  │
│   │  Flash)  │    │   Flash)  │    │    Stage A       │  │
│   └─────────┘    └───────────┘    └────────┬─────────┘  │
│        ▲              ▲                     │            │
│        │              │              ┌──────▼─────────┐  │
│        │              │              │  LLM Style     │  │
│        │        ┌─────┴──────┐       │   Critic       │  │
│        │        │  Retry     │       │   (Flash)      │  │
│        │        │  (max 2)   │◂──────│   Stage B      │  │
│        │        └────────────┘       └──────┬─────────┘  │
│        │                                     │           │
│   ┌────┴────────────────────────────────────▼──────┐    │
│   │              Accept → Dialogue History          │    │
│   └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
         │                                    │
    ┌────▼────┐                         ┌────▼────┐
    │ ChromaDB│                         │ FastAPI │
    │ Golden  │                         │   +     │
    │ Lines   │                         │ WebSocket│
    └─────────┘                         └────┬────┘
                                             │
                                        ┌────▼────┐
                                        │ Split-  │
                                        │ Screen  │
                                        │   UI    │
                                        └─────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
cd gen-Clash
pip install -r requirements.txt
```

### 2. Configure Gemini API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

*(Optional model overrides default to `gemini-3.5-flash-lite` with fallback to `gemini-3.1-flash-lite`)*.

### 3. Initialize Vector Store

```bash
python -m writers_room.run init-store
```

### 4. Validate Voice Profiles

```bash
python -m writers_room.run validate-profiles
```

### 5. Generate a Scene (CLI)

```bash
python -m writers_room.run generate \
  --premise "A smuggler confronts a geneticist about stolen cargo in an abandoned warehouse while a teenage hacker watches from the rafters" \
  --location "Abandoned Warehouse — Dockside District" \
  --characters merc_vance dr_lena_soros tommy_k \
  --beats 8 \
  -o scene.fountain
```

### 6. Start the Interactive Web UI

```bash
python -m writers_room.run serve
```
Open your browser at:
👉 **[http://localhost:8000/](http://localhost:8000/)** *(or `http://localhost:8000/ui`)*

---

## Character Voice Matrix

| Character | Archetype | Max Words | Contraction Rate | Dialect Quirks | Forbidden Lexicon |
|-----------|-----------|-----------|-----------------|----------------|-------------------|
| ⚓ **Merc Vance** | Cynical Dockside Smuggler | 14 | High (0.85) | Drops pronouns, maritime metaphors, short imperatives | *furthermore, comprehend, traumatized, basically, please* |
| 🧬 **Dr. Lena Soros** | Cold Corporate Geneticist | 25 | Low (0.10) | Passive voice, Latinate syntax, clinical detachment | *ain't, gotta, reckon, cops, gut feeling, okay* |
| 💻 **Tommy K** | Hyperactive Teen Hacker | 10 | Medium (0.55) | Stream-of-consciousness, stutter fragments, gaming slang | *furthermore, optimal, protocol, paradigm, esteemed* |

---

## Two-Stage Voice Linter

Every generated line passes through two gates before entering the screenplay:

### Stage A — Deterministic Linguistic Checks
| Check | Purpose |
|-------|---------|
| **Sentence Length** | Ensures each sentence stays within character-specific maximum word limits. |
| **Forbidden Lexicon** | Word-boundary regex checks preventing characters from using out-of-register vocabulary. |
| **Contraction Rate** | Verifies ratio of contractions matches the character's social register. |
| **Phonetic Markers** | Requires character dialect markers (`'em`, `ain't`, `oughta`, `literally`) in extended sentences. |
| **AI Cadence Tells** | Catches generic LLM habits: *"I understand"*, *"It's important to"*, therapy-speak, and semicolon balance. |

### Stage B — LLM Style Critic
Compares candidate lines against retrieved canonical golden lines:
- Therapy-speak and unnatural hedging detection
- Register and tone dissonance
- Rhythm and flow divergence from canonical examples

---

## Dynamic Dialect Adaptation (Milestone 3)

Characters automatically shift speech patterns in real time:

- **Scene Stress Escalation**: As scene tension escalates, Merc's sentences compress further, Dr. Soros's clinical facade slips into defensive jargon, and Tommy's speech becomes fragmented.
- **Peer vs. Authority Code-Switching**: Characters adapt their speech depending on their conversation partner (e.g., Tommy suppresses hacker slang when directly interrogated by Dr. Soros).
- **Interactive UI Dialect Sliders**: Users can dynamically manipulate character slang density, sentence length, and aggression during live generation via the WebSocket control pane.

---

## Project Structure

```
writers_room/
├── __init__.py              # Package root
├── __main__.py              # python -m writers_room
├── run.py                   # CLI entry point (serve, generate, init-store, validate-profiles)
├── config.py                # Central configuration (.env auto-load, models, linter thresholds)
├── agents/
│   ├── director.py          # Scene Director (Gemini 3.5 Flash-Lite + resilient fallback)
│   ├── character.py         # Character Agent Factory (isolated context + RAG injection)
│   └── dialect_adapter.py   # Dynamic dialect adaptation (stress curves & code-switching)
├── data/
│   ├── schemas.py           # Pydantic v2 data models
│   ├── voice_profiles/      # Character voice constraint JSONs
│   │   ├── merc_vance.json
│   │   ├── dr_lena_soros.json
│   │   └── tommy_k.json
│   └── golden_lines/        # Canonical voice exemplars (30 per character = 90 lines)
│       ├── merc_vance_golden.json
│       ├── dr_lena_soros_golden.json
│       └── tommy_k_golden.json
├── graph/
│   ├── scene_graph.py       # LangGraph cyclic state machine with retry loops
│   └── formatter.py         # Fountain screenplay formatter
├── linter/
│   ├── voice_linter.py      # Stage A: Deterministic linguistic checks (regex & rules)
│   └── critic.py            # Stage B: LLM style critic (golden-line comparison)
├── memory/
│   └── vector_store.py      # ChromaDB golden-line RAG with resilient JSON fallback
├── api/
│   └── server.py            # FastAPI REST & bi-directional WebSocket server
├── ui/
│   ├── index.html           # Split-screen screenplay editor interface
│   ├── styles.css           # Cinematic dark theme styling
│   └── app.js               # WebSocket streaming client + real-time dialect sliders
├── tests/
│   ├── test_linter.py       # Deterministic linguistic checker unit tests
│   ├── test_dialect_adapter.py # Stress and code-switching tests
│   ├── test_scene_graph.py  # LangGraph cyclic state machine tests
│   ├── test_vector_store.py # Golden line RAG retrieval tests
│   ├── test_api_server.py   # FastAPI endpoints & static file tests
│   └── test_voice_separation.py # Embedding cluster separation benchmark
└── requirements.txt
```

---

## Running the Automated Test Suite

```bash
# Run all 58 tests across linter, dialect adapter, vector store, and scene graph
pytest

# Run deterministic voice linter tests specifically
pytest writers_room/tests/test_linter.py -v

# Run embedding cluster separation benchmark
pytest writers_room/tests/test_voice_separation.py -v
```

---

## License

MIT
