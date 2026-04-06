"""
Deterministic re-ranker — NO LLM, NO network, pure computation.

Heuristics:
  1. Token-overlap score (Jaccard similarity between query and snippet).
  2. Exact phrase boost (case-insensitive substring match).
  3. Title / heading boost (token overlap with query).

All operations are deterministic and testable.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from app.services.retrieval.types import ScoredChunk

logger = logging.getLogger(__name__)


@runtime_checkable
class ReRanker(Protocol):
    """Re-ranking interface."""

    async def rerank(
        self,
        *,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Re-rank *chunks* for *query* and return the top *top_k*."""
        ...


class DeterministicReRanker:
    """
    Pure-computation re-ranker.

    Scoring formula::

        rerank_score = BASE_WEIGHT * hybrid_score
                     + OVERLAP_WEIGHT * jaccard(query_tokens, snippet_tokens)
                     + PHRASE_BOOST   * (1 if query in snippet)
                     + TITLE_BOOST    * title_overlap
                     + HEADING_BOOST  * heading_overlap

    All weights are class-level constants — deterministic, reproducible.
    """

    BASE_WEIGHT: float = 0.55
    OVERLAP_WEIGHT: float = 0.25
    PHRASE_BOOST: float = 0.10
    TITLE_BOOST: float = 0.06
    HEADING_BOOST: float = 0.04

    __slots__ = ()

    async def rerank(
        self,
        *,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        query_tokens = self._tokenize(query)
        scored: list[ScoredChunk] = []

        for chunk in chunks:
            snippet_tokens = self._tokenize(chunk.snippet)

            # 1. Jaccard token overlap
            if query_tokens and snippet_tokens:
                intersection = len(query_tokens & snippet_tokens)
                union = len(query_tokens | snippet_tokens)
                overlap = intersection / union if union else 0.0
            else:
                overlap = 0.0

            # 2. Exact phrase boost
            phrase_match = 1.0 if (
                query.strip()
                and query.lower() in (chunk.snippet or "").lower()
            ) else 0.0

            # 3. Title token overlap
            title_overlap = self._field_overlap(query_tokens, chunk.title)

            # 4. Heading token overlap
            heading_overlap = self._field_overlap(query_tokens, chunk.heading)

            new_score = (
                self.BASE_WEIGHT * chunk.score
                + self.OVERLAP_WEIGHT * overlap
                + self.PHRASE_BOOST * phrase_match
                + self.TITLE_BOOST * title_overlap
                + self.HEADING_BOOST * heading_overlap
            )

            scored.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    version_id=chunk.version_id,
                    chunk_index=chunk.chunk_index,
                    score=round(new_score, 8),
                    source="rerank",
                    snippet=chunk.snippet,
                    title=chunk.title,
                    heading=chunk.heading,
                )
            )

        scored.sort(key=lambda c: (-c.score, c.chunk_id))
        result = scored[:top_k]

        logger.info(
            "reranker.rerank input=%d output=%d top_k=%d",
            len(chunks), len(result), top_k,
        )
        return result

    @staticmethod
    def _tokenize(text: str | None) -> set[str]:
        """Lowercase word tokenisation with punctuation stripping."""
        if not text:
            return set()
        return {
            w.strip(".,;:!?\"'()[]{}") for w in text.lower().split()
        } - {""}

    @staticmethod
    def _field_overlap(query_tokens: set[str], field: str | None) -> float:
        """Fraction of query tokens found in *field*."""
        if not field or not query_tokens:
            return 0.0
        field_tokens = {
            w.strip(".,;:!?\"'()[]{}") for w in field.lower().split()
        } - {""}
        if not field_tokens:
            return 0.0
        return len(query_tokens & field_tokens) / len(query_tokens)
