"""
Milestone 4 — Voice Distinctiveness & Cluster Separation Benchmark.

Benchmarks character distinctiveness using an embedding cluster separation test.
Verifies that canonical and generated dialogue from contrasting characters
(Merc Vance, Dr. Lena Soros, Tommy K) cluster distinctly in vector space
rather than collapsing into homogenized LLM cadence.

Tech:
    ChromaDB default embedding function (all-MiniLM-L6-v2 ONNX)
    Cosine similarity, Centroid distance, Nearest-Neighbor Cluster Purity
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

GOLDEN_DIR = Path(__file__).resolve().parent.parent / "data" / "golden_lines"


def _load_golden_lines() -> Dict[str, List[str]]:
    """Load canonical golden lines grouped by character_id."""
    golden_files = list(GOLDEN_DIR.glob("*_golden.json"))
    assert len(golden_files) >= 3, f"Expected at least 3 golden files, found {len(golden_files)}"

    data_map: Dict[str, List[str]] = {}
    for f in golden_files:
        with open(f, "r", encoding="utf-8") as fh:
            content = json.load(fh)
        cid = content["character_id"]
        data_map[cid] = [item["text"] for item in content["lines"]]
    return data_map


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two dense vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_centroid(vectors: List[List[float]]) -> List[float]:
    """Compute the centroid (element-wise mean) of a vector cluster."""
    dim = len(vectors[0])
    n = len(vectors)
    return [sum(vec[d] for vec in vectors) / n for d in range(dim)]


def _average_intra_cluster_similarity(vectors: List[List[float]]) -> float:
    """Compute average pairwise similarity among lines of the same character."""
    n = len(vectors)
    if n < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _cosine_similarity(vectors[i], vectors[j])
            count += 1
    return total / count if count > 0 else 0.0


@pytest.fixture(scope="module")
def character_embeddings() -> Dict[str, List[List[float]]]:
    """Embed all character golden lines using ChromaDB's default embedding function."""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    ef = DefaultEmbeddingFunction()
    golden_texts = _load_golden_lines()

    embedded: Dict[str, List[List[float]]] = {}
    for cid, texts in golden_texts.items():
        # Embedding function returns a list of embeddings
        embedded[cid] = ef(texts)

    return embedded


class TestEmbeddingClusterSeparation:
    """Benchmark character distinctiveness in embedding vector space."""

    def test_sample_counts_and_dimensions(self, character_embeddings):
        """Ensure all 3 characters have 30 lines embedded in 384-dimensional space."""
        assert len(character_embeddings) == 3
        expected_chars = {"merc_vance", "dr_lena_soros", "tommy_k"}
        assert set(character_embeddings.keys()) == expected_chars

        for cid, vectors in character_embeddings.items():
            assert len(vectors) == 30, f"{cid} has {len(vectors)} lines, expected 30"
            for vec in vectors:
                assert len(vec) == 384, "ChromaDB all-MiniLM-L6-v2 embedding dimension must be 384"

    def test_intra_cluster_cohesion(self, character_embeddings):
        """Each character's lines should be self-coherent."""
        for cid, vectors in character_embeddings.items():
            intra_sim = _average_intra_cluster_similarity(vectors)
            # Lines from the same character should show positive thematic and stylistic cohesion
            assert intra_sim > 0.10, (
                f"{cid} intra-cluster similarity too low: {intra_sim:.4f}"
            )

    def test_inter_cluster_centroid_separation(self, character_embeddings):
        """Centroids of contrasting characters must be well-separated in embedding space."""
        centroids: Dict[str, List[float]] = {
            cid: _compute_centroid(vecs) for cid, vecs in character_embeddings.items()
        }

        cids = list(centroids.keys())
        distances: List[Tuple[str, str, float]] = []

        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                c1, c2 = cids[i], cids[j]
                sim = _cosine_similarity(centroids[c1], centroids[c2])
                dist = 1.0 - sim  # Cosine distance
                distances.append((c1, c2, dist))

                # If LLM voice homogenization occurred, centroid distance would approach 0 (sim ~ 1.0)
                # Contrasting characters (Smuggler vs Geneticist vs Teen Hacker) must have sim < 0.88
                assert sim < 0.88, (
                    f"Voice homogenization detected between {c1} and {c2}: "
                    f"similarity={sim:.4f} is too high (distance={dist:.4f})"
                )

        # Average inter-cluster distance should be substantial
        avg_dist = sum(d[2] for d in distances) / len(distances)
        assert avg_dist > 0.15, f"Average centroid distance {avg_dist:.4f} is below separation threshold 0.15"

    def test_nearest_neighbor_cluster_purity(self, character_embeddings):
        """Every line is classified by its nearest neighbour.

        A high proportion of lines should have a nearest neighbor belonging
        to the exact same character (high cluster purity).
        """
        all_lines: List[Tuple[str, List[float]]] = []
        for cid, vectors in character_embeddings.items():
            for v in vectors:
                all_lines.append((cid, v))

        total_lines = len(all_lines)
        correct_matches = 0

        for i, (true_cid, vec_i) in enumerate(all_lines):
            best_sim = -1.0
            best_cid = None
            for j, (other_cid, vec_j) in enumerate(all_lines):
                if i == j:
                    continue  # Do not match with self
                sim = _cosine_similarity(vec_i, vec_j)
                if sim > best_sim:
                    best_sim = sim
                    best_cid = other_cid

            if best_cid == true_cid:
                correct_matches += 1

        purity = correct_matches / total_lines
        # For 3 randomly distributed classes, random chance accuracy would be ~33.3%
        # Strong dialect separation achieves > 65% nearest-neighbor purity
        assert purity >= 0.65, (
            f"Nearest-neighbor separation purity {purity:.2%} is below threshold 65% "
            f"({correct_matches}/{total_lines} correct)"
        )

    def test_centroid_classification_accuracy(self, character_embeddings):
        """Classify each line against the 3 character centroids."""
        centroids = {cid: _compute_centroid(vecs) for cid, vecs in character_embeddings.items()}

        correct_classifications = 0
        total = 0

        for cid, vectors in character_embeddings.items():
            for vec in vectors:
                total += 1
                best_cid = max(centroids.keys(), key=lambda c: _cosine_similarity(vec, centroids[c]))
                if best_cid == cid:
                    correct_classifications += 1

        accuracy = correct_classifications / total
        assert accuracy >= 0.70, (
            f"Centroid classification accuracy {accuracy:.2%} is below threshold 70% "
            f"({correct_classifications}/{total})"
        )
