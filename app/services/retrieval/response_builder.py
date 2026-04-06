"""
Response builder — converts ``ScoredChunk`` → ``QueryResult``.

Produces snippets with simple deterministic highlighting (wrapping
matched query terms).  No logging of raw text.

Step 5: Accepts optional ``selection_meta`` map for debug metadata
and ``source_document_id`` mapping for citation fidelity.
"""
from __future__ import annotations

import re

from app.services.retrieval.types import QueryResult, ScoredChunk


class ResponseBuilder:
    """
    Stateless converter from internal ``ScoredChunk`` to API-facing
    ``QueryResult``.
    """

    __slots__ = ()

    @staticmethod
    def build(
        chunks: list[ScoredChunk],
        query: str,
        *,
        selection_meta: dict[int, dict] | None = None,
        source_doc_map: dict[int, int] | None = None,
    ) -> list[QueryResult]:
        """
        Convert scored chunks to query results with highlights.

        Args:
            chunks: Reranked/consolidated scored chunks.
            query: Original query text for highlight extraction.
            selection_meta: Optional mapping from document_id to
                debug metadata dict (populated when debug mode active).
            source_doc_map: Optional mapping from document_id to
                source/original document_id (for citation fidelity).
        """
        query_tokens = _tokenize(query)
        results: list[QueryResult] = []

        for c in chunks:
            highlights = _extract_highlights(c.snippet, query_tokens)

            # Source fidelity: map to original doc if available
            source_doc_id = (
                source_doc_map.get(c.document_id)
                if source_doc_map
                else None
            )

            # Debug metadata: attach if available
            debug = (
                selection_meta.get(c.document_id)
                if selection_meta
                else None
            )

            results.append(
                QueryResult(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    score=c.score,
                    snippet=c.snippet,
                    highlights=tuple(highlights),
                    title=c.title,
                    heading=c.heading,
                    source_document_id=source_doc_id,
                    debug_meta=debug,
                )
            )

        return results


# ── private helpers ───────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lowercase word tokens with punctuation stripped."""
    return {
        w.strip(".,;:!?\"'()[]{}") for w in text.lower().split()
    } - {""}


def _extract_highlights(
    snippet: str,
    query_tokens: set[str],
) -> list[str]:
    """
    Return the list of *unique* words in *snippet* that match any
    query token (case-insensitive).  Preserves original casing.
    """
    if not snippet or not query_tokens:
        return []

    seen: set[str] = set()
    highlights: list[str] = []
    for word in snippet.split():
        clean = word.strip(".,;:!?\"'()[]{}").lower()
        if clean in query_tokens and clean not in seen:
            seen.add(clean)
            highlights.append(word.strip(".,;:!?\"'()[]{}"))

    return highlights
