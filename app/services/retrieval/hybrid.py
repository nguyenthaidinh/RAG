"""
Hybrid search strategy — deterministic score fusion.

Combines BM25 (lexical) and vector (semantic) results using min-max
normalisation and a weighted linear combination.  No randomness, no LLM,
no network calls.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Literal

from app.services.retrieval.types import ScoredChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridConfig:
    """Tuning knobs for hybrid score fusion."""

    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    normalize: Literal["minmax"] = "minmax"
    threshold: float = 0.0


def _minmax_normalize(scores: list[float]) -> list[float]:
    """
    Normalize *scores* to [0, 1] using min-max scaling.

    If all scores are identical (max == min), return 1.0 for every entry
    to avoid division by zero.
    """
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    span = hi - lo
    if span == 0.0:
        return [1.0] * len(scores)
    return [(s - lo) / span for s in scores]


class HybridStrategy:
    """
    Merges vector and BM25 result sets with deterministic fusion.

    Steps:
      1. Min-max normalise scores within each source.
      2. Index results by ``chunk_id``.
      3. Compute ``combined = w_vec * vec_score + w_bm25 * bm25_score``
         (missing scores treated as 0).
      4. Drop results below ``threshold``.
      5. Sort descending by combined score.
    """

    __slots__ = ("_cfg",)

    def __init__(self, config: HybridConfig | None = None) -> None:
        self._cfg = config or HybridConfig()

    def merge(
        self,
        *,
        vector_results: list[ScoredChunk],
        bm25_results: list[ScoredChunk],
    ) -> list[ScoredChunk]:
        # ── normalise ─────────────────────────────────────────────
        vec_norm = _minmax_normalize([c.score for c in vector_results])
        bm25_norm = _minmax_normalize([c.score for c in bm25_results])

        # ── build lookup: chunk_id → (vec_score, bm25_score, chunk)
        combined: dict[int, dict] = {}

        for chunk, norm_score in zip(vector_results, vec_norm):
            combined[chunk.chunk_id] = {
                "vec": norm_score,
                "bm25": 0.0,
                "chunk": chunk,
            }

        for chunk, norm_score in zip(bm25_results, bm25_norm):
            if chunk.chunk_id in combined:
                combined[chunk.chunk_id]["bm25"] = norm_score
            else:
                combined[chunk.chunk_id] = {
                    "vec": 0.0,
                    "bm25": norm_score,
                    "chunk": chunk,
                }

        # ── fuse ──────────────────────────────────────────────────
        w_vec = self._cfg.vector_weight
        w_bm25 = self._cfg.bm25_weight
        threshold = self._cfg.threshold

        merged: list[ScoredChunk] = []
        for entry in combined.values():
            score = w_vec * entry["vec"] + w_bm25 * entry["bm25"]
            if score < threshold:
                continue
            base: ScoredChunk = entry["chunk"]
            merged.append(
                ScoredChunk(
                    chunk_id=base.chunk_id,
                    document_id=base.document_id,
                    version_id=base.version_id,
                    chunk_index=base.chunk_index,
                    score=round(score, 8),
                    source="hybrid",
                    snippet=base.snippet,
                    title=base.title,
                    heading=base.heading,
                )
            )

        # ── sort descending by score ──────────────────────────────
        merged.sort(key=lambda c: c.score, reverse=True)

        logger.info(
            "hybrid.merge vector_in=%d bm25_in=%d merged=%d threshold=%.3f",
            len(vector_results), len(bm25_results), len(merged),
            threshold,
        )
        return merged
