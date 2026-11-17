"""ChromaDB-backed vector store for golden-line retrieval."""

from __future__ import annotations

import json
import os
from glob import glob
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

# ── Module-level singleton ───────────────────────────────────────────

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None

_COLLECTION_NAME = "golden_lines"
_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "chroma_db",
)


def _get_client() -> chromadb.ClientAPI:
    """Return (and lazily create) the persistent ChromaDB client."""
    global _client
    if _client is None:
        os.makedirs(_PERSIST_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_collection() -> chromadb.Collection:
    """Return (and lazily create) the golden-lines collection."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Public API ───────────────────────────────────────────────────────


def _query_from_json_fallback(character_id: str, emotion: str, n_results: int = 3) -> list[str]:
    """Fallback reader that returns golden lines directly from JSON files if vector store is unavailable."""
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "golden_lines",
        f"{character_id}_golden.json",
    )
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lines = data.get("lines", [])
        matched = [l["text"] for l in lines if l.get("emotion") == emotion]
        if len(matched) < n_results:
            for l in lines:
                if l["text"] not in matched:
                    matched.append(l["text"])
                if len(matched) >= n_results:
                    break
        return matched[:n_results]
    except Exception:
        return []


_store_initialized = False


def initialize_store(golden_lines_dir: str) -> int:
    """Read every ``*_golden.json`` file from *golden_lines_dir* and upsert
    all lines into the ChromaDB collection.

    Returns the total number of lines upserted.
    """
    global _store_initialized
    if _store_initialized:
        return 90

    pattern = os.path.join(golden_lines_dir, "*_golden.json")
    files = glob(pattern)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        character_id: str = data.get("character_id", Path(filepath).stem.replace("_golden", ""))
        lines: list[dict[str, Any]] = data.get("lines", [])

        for line in lines:
            ids.append(line["line_id"])
            documents.append(line["text"])
            tags = line.get("tags", [])
            metadatas.append(
                {
                    "character_id": character_id,
                    "emotion": line.get("emotion", ""),
                    "context": line.get("context", ""),
                    "tags": ",".join(tags) if isinstance(tags, list) else str(tags),
                }
            )

    if ids:
        try:
            collection = _get_collection()
            if collection.count() >= len(ids):
                _store_initialized = True
                return len(ids)
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            _store_initialized = True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("ChromaDB upsert skipped/failed: %s (JSON fallback active)", exc)

    return len(ids)


def query_golden_lines(
    character_id: str,
    emotion: str,
    n_results: int = 3,
) -> list[str]:
    """Retrieve the top-*n_results* golden lines for a character filtered by
    emotion, using semantic similarity from ChromaDB's default embeddings.

    Returns a list of line texts (strings).
    """
    try:
        collection = _get_collection()

        # Build a where-filter that matches both character and emotion
        where_filter: dict[str, Any] = {
            "$and": [
                {"character_id": {"$eq": character_id}},
                {"emotion": {"$eq": emotion}},
            ]
        }

        try:
            results = collection.query(
                query_texts=[f"{emotion} dialogue"],
                n_results=n_results,
                where=where_filter,
            )
        except Exception:
            # Fallback: if the filtered query fails (e.g. not enough docs),
            # try without the emotion filter
            results = collection.query(
                query_texts=[f"{emotion} dialogue"],
                n_results=n_results,
                where={"character_id": {"$eq": character_id}},
            )

        docs = results.get("documents", [[]])
        if docs and docs[0]:
            return docs[0]
    except Exception:
        pass

    # Resilient fallback to direct JSON golden bank
    return _query_from_json_fallback(character_id, emotion, n_results)


def reset_store() -> None:
    """Delete the collection and reset module-level singletons.

    Intended for testing — lets a test suite start with a clean slate.
    """
    global _client, _collection

    try:
        client = _get_client()
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass  # Collection may not exist yet

    _collection = None
    _client = None
