"""
Entry point for the Writers' Room application.

Usage:
    # Start the API server (serves both API and UI):
    python -m writers_room.run serve

    # Run a scene from the CLI:
    python -m writers_room.run generate --premise "Two smugglers argue..." --location "Dockside warehouse" --characters merc_vance dr_lena_soros --beats 8

    # Initialize the vector store with golden lines:
    python -m writers_room.run init-store

    # Validate all voice profiles:
    python -m writers_room.run validate-profiles
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


def cmd_serve(args):
    """Start the FastAPI server."""
    import uvicorn
    from writers_room.config import API_HOST, API_PORT

    host = args.host or API_HOST
    port = args.port or API_PORT

    display_host = "localhost" if host in ("0.0.0.0", "127.0.0.1") else host
    print(f"\n  🎬 The Writers' Room — Server running at:")
    print(f"  👉 http://{display_host}:{port}/")
    print(f"  👉 http://{display_host}:{port}/ui\n")

    uvicorn.run(
        "writers_room.api.server:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info",
    )


def cmd_init_store(args):
    """Initialize ChromaDB with golden lines."""
    from writers_room.config import GOLDEN_LINES_DIR
    from writers_room.memory.vector_store import initialize_store

    print("🔄 Initializing vector store with golden lines...")
    count = initialize_store(str(GOLDEN_LINES_DIR))
    print(f"✅ Loaded {count} golden lines into ChromaDB.")


def cmd_validate_profiles(args):
    """Validate all character voice profiles."""
    from writers_room.config import VOICE_PROFILES_DIR
    from writers_room.data.schemas import CharacterVoiceProfile

    profiles_dir = Path(VOICE_PROFILES_DIR)
    errors = []

    for profile_path in profiles_dir.glob("*.json"):
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            CharacterVoiceProfile(**data)
            print(f"  ✅ {profile_path.name}")
        except Exception as e:
            errors.append((profile_path.name, str(e)))
            print(f"  ❌ {profile_path.name}: {e}")

    if errors:
        print(f"\n❌ {len(errors)} profile(s) failed validation.")
        sys.exit(1)
    else:
        print(f"\n✅ All profiles valid.")


def cmd_generate(args):
    """Generate a scene from the command line."""

    async def _run():
        from writers_room.config import VOICE_PROFILES_DIR, GOLDEN_LINES_DIR
        from writers_room.graph.scene_graph import build_scene_graph
        from writers_room.graph.formatter import format_screenplay
        from writers_room.memory.vector_store import initialize_store
        from writers_room.memory import vector_store as vs_module
        from writers_room.agents.director import DirectorAgent
        from writers_room.agents.character import CharacterAgent

        # Initialize store if needed
        initialize_store(str(GOLDEN_LINES_DIR))

        # Load character profiles
        profiles = {}
        for cid in args.characters:
            profile_path = Path(VOICE_PROFILES_DIR) / f"{cid}.json"
            if not profile_path.exists():
                print(f"❌ Profile not found: {profile_path}")
                sys.exit(1)
            with open(profile_path, "r", encoding="utf-8") as f:
                profiles[cid] = json.load(f)

        # Create runtime agents
        director = DirectorAgent()
        char_pool = {}
        for cid, profile in profiles.items():
            char_pool[cid] = CharacterAgent(
                character_id=cid,
                voice_profile=profile,
                vector_store=vs_module,
            )

        # Build and run graph
        graph = build_scene_graph()

        initial_state = {
            "scene_id": "cli_scene_001",
            "premise": args.premise,
            "location": args.location,
            "active_characters": args.characters,
            "character_profiles": profiles,
            "beats": [],
            "dialogue": [],
            "current_beat": None,
            "current_line": None,
            "current_adapted_profile": None,
            "lint_result": None,
            "critic_verdict": None,
            "retry_count": 0,
            "max_retries": 2,
            "max_beats": args.beats,
            "errors": [],
            "is_complete": False,
            "_director": director,
            "_character_pool": char_pool,
            "_event_queue": None,
        }

        print(f"\n🎬 Generating scene: {args.premise}")
        print(f"📍 Location: {args.location}")
        print(f"🎭 Characters: {', '.join(args.characters)}")
        print(f"🎯 Beats: {args.beats}\n")
        print("=" * 60)

        final_state = await graph.ainvoke(initial_state)

        screenplay = format_screenplay(final_state)
        print(screenplay)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(screenplay)
            print(f"\n💾 Saved to {args.output}")

    asyncio.run(_run())


def main():
    parser = argparse.ArgumentParser(
        description="The Writers' Room — Multi-Agent Dialect & Voice Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--reload", action="store_true")

    # init-store
    subparsers.add_parser("init-store", help="Initialize the vector store")

    # validate-profiles
    subparsers.add_parser("validate-profiles", help="Validate voice profiles")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate a scene from CLI")
    gen_parser.add_argument("--premise", required=True, help="Scene premise")
    gen_parser.add_argument("--location", required=True, help="Scene location")
    gen_parser.add_argument("--characters", nargs="+", required=True, help="Character IDs")
    gen_parser.add_argument("--beats", type=int, default=10, help="Number of beats")
    gen_parser.add_argument("--output", "-o", help="Output file path")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "init-store":
        cmd_init_store(args)
    elif args.command == "validate-profiles":
        cmd_validate_profiles(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

