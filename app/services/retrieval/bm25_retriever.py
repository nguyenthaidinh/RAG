"""
BM25 (lexical) retriever.

Wraps a ``BM25Repository`` and converts raw FTS hits to ``ScoredChunk``
objects.  Document-level results are mapped to chunk_index=0.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from app.services.retrieval.bm25_repo import BM25Repository, NullBM25Repository
from app.services.retrieval.types import ScoredChunk, make_chunk_id

logger = logging.getLogger(__name__)


@runtime_checkable
class BM25Retriever(Protocol):
    """Retriever interface for BM25 / lexical search."""

    async def search(
        self,
        *,
        tenant_id: str,
        query_text: str,
        limit: int,
        allowed_doc_ids: set[int],
    ) -> list[ScoredChunk]:
        """Search for chunks matching *query_text* via BM25."""
        ...


class DefaultBM25Retriever:
    """
    Production BM25 retriever backed by PostgreSQL FTS.

    Converts ``BM25Hit`` (document-level) to ``ScoredChunk`` with
    ``chunk_index=0`` and ``source="bm25"``.
    """

    __slots__ = ("_repo",)

    def __init__(self, repo: BM25Repository | None = None) -> None:
        self._repo = repo or NullBM25Repository()

    async def search(
        self,
        *,
        tenant_id: str,
        query_text: str,
        limit: int,
        allowed_doc_ids: set[int],
    ) -> list[ScoredChunk]:
        hits = await self._repo.search(
            tenant_id=tenant_id,
            query_text=query_text,
            limit=limit,
            allowed_doc_ids=list(allowed_doc_ids),
        )

        chunks: list[ScoredChunk] = []
        for h in hits:
            chunks.append(
                ScoredChunk(
                    chunk_id=make_chunk_id(h.document_id, 0),
                    document_id=h.document_id,
                    version_id=h.version_id,
                    chunk_index=0,
                    score=h.rank,
                    source="bm25",
                    snippet=h.headline,
                    title=h.title,
                )
            )

        logger.info(
            "bm25_retriever.search tenant_id=%s hits=%d",
            tenant_id, len(chunks),
        )
        return chunks


class NullBM25Retriever:
    """No-op retriever — returns empty results."""

    async def search(self, **kw) -> list[ScoredChunk]:
        return []
