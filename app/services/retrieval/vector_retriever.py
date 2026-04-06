"""
Vector similarity retriever.

Defines a ``VectorSearchBackend`` protocol (internal to the retrieval
package) that decouples the retriever from specific vector DB vendors.
The Phase 3.2 ``VectorIndex`` interface is NOT modified.

Implementations:
  * ``NullVectorSearchBackend`` — returns ``[]`` (default / CI).
  * ``QdrantVectorSearchBackend`` — uses ``qdrant-client`` if installed.
  * ``PgVectorSearchBackend`` — PostgreSQL pgvector via SQLAlchemy async.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.services.retrieval.types import ScoredChunk, VectorFilter, make_chunk_id

logger = logging.getLogger(__name__)


# ── raw hit from vector DB (internal) ─────────────────────────────────

@dataclass(frozen=True)
class RawVectorHit:
    """Minimal result returned by a search backend."""

    document_id: int
    version_id: str
    chunk_index: int
    score: float
    payload: dict          # arbitrary metadata from vector store


# ── search backend protocol (internal) ────────────────────────────────

class VectorSearchBackend(Protocol):
    """
    Internal protocol for raw vector similarity search.

    Implementations talk to a specific vector DB and return
    ``RawVectorHit`` objects.  They MUST enforce ``tenant_id`` and
    ``allowed_doc_ids`` filtering at the vector-DB level.
    """

    async def search(
        self,
        *,
        query_vector: list[float],
        limit: int,
        tenant_id: str,
        allowed_doc_ids: frozenset[int],
    ) -> list[RawVectorHit]:
        """Search for nearest neighbours filtered by tenant + doc IDs."""
        ...


# ── null backend (default / CI) ──────────────────────────────────────

class NullVectorSearchBackend:
    """No-op backend — always returns empty results."""

    __slots__ = ()

    async def search(self, **kw) -> list[RawVectorHit]:
        return []


# ── Qdrant backend ───────────────────────────────────────────────────

class QdrantVectorSearchBackend:
    """
    Qdrant-backed vector search.

    Uses the same collection and point-ID scheme as
    ``app.services.vector_index.QdrantIndex`` (Phase 3.2).

    Requires optional dependency: ``qdrant-client``.
    """

    __slots__ = ("_client", "_collection")

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
    ) -> None:
        try:
            from qdrant_client import AsyncQdrantClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "QdrantVectorSearchBackend requires 'qdrant-client'. "
                "Install with: pip install qdrant-client"
            ) from exc

        from app.core.config import settings

        self._collection = collection or settings.QDRANT_COLLECTION
        _url = url or settings.QDRANT_URL
        _key = (api_key or settings.QDRANT_API_KEY) or None
        self._client = AsyncQdrantClient(url=_url, api_key=_key)

    async def search(
        self,
        *,
        query_vector: list[float],
        limit: int,
        tenant_id: str,
        allowed_doc_ids: frozenset[int],
    ) -> list[RawVectorHit]:
        from qdrant_client.models import (  # type: ignore[import-untyped]
            FieldCondition,
            Filter,
            MatchAny,
            MatchValue,
        )

        results = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=limit,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id),
                    ),
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=list(allowed_doc_ids)),
                    ),
                ],
            ),
        )

        hits: list[RawVectorHit] = []
        for point in results:
            payload = point.payload or {}
            hits.append(
                RawVectorHit(
                    document_id=int(payload.get("document_id", 0)),
                    version_id=str(payload.get("version_id", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    score=float(point.score),
                    payload=payload,
                )
            )

        logger.info(
            "qdrant_backend.search tenant_id=%s hits=%d limit=%d",
            tenant_id, len(hits), limit,
        )
        return hits


# ── pgvector backend ─────────────────────────────────────────────────

class PgVectorSearchBackend:
    """
    pgvector-backed vector search using SQLAlchemy 2.0 async.

    Queries the ``document_vectors`` table directly in PostgreSQL,
    joining ``documents`` for defense-in-depth tenant isolation.

    Uses a short-lived session from the provided session factory
    so the connection is released immediately after the query.
    """

    __slots__ = ("_session_factory",)

    def __init__(self, session_factory=None) -> None:
        if session_factory is None:
            from app.db.session import AsyncSessionLocal
            session_factory = AsyncSessionLocal
        self._session_factory = session_factory

    async def search(
        self,
        *,
        query_vector: list[float],
        limit: int,
        tenant_id: str,
        allowed_doc_ids: frozenset[int],
    ) -> list[RawVectorHit]:
        if not allowed_doc_ids:
            logger.warning(
                "DEBUG pgvector_search SKIPPED tenant_id=%s allowed_doc_ids EMPTY",
                tenant_id,
            )
            return []

        logger.warning(
            "DEBUG pgvector_search tenant_id=%s allowed_count=%d allowed_doc_ids=%s limit=%d",
            tenant_id,
            len(allowed_doc_ids),
            sorted(list(allowed_doc_ids)),
            limit,
        )

        from sqlalchemy import text

        # Convert Python list[float] → pgvector literal string
        # Same pattern used by PgVectorRepository.upsert_batch
        emb_str = "[" + ",".join(str(v) for v in query_vector) + "]"

        sql = text(
            """
            SELECT
                v.document_id   AS document_id,
                v.version_id    AS version_id,
                v.chunk_index   AS chunk_index,
                v.chunk_text    AS chunk_text,
                (v.embedding <-> CAST(:qvec AS vector)) AS distance
            FROM document_vectors v
            JOIN documents d ON d.id = v.document_id
            WHERE d.tenant_id = :tenant_id
              AND v.document_id = ANY(:allowed_doc_ids)
            ORDER BY v.embedding <-> CAST(:qvec AS vector)
            LIMIT :limit
            """
        )

        async with self._session_factory() as db:
            result = await db.execute(
                sql,
                {
                    "qvec": emb_str,
                    "tenant_id": tenant_id,
                    "allowed_doc_ids": list(allowed_doc_ids),
                    "limit": limit,
                },
            )
            rows = result.fetchall()

        hits: list[RawVectorHit] = []
        for row in rows:
            distance = float(row.distance)
            chunk_text = getattr(row, "chunk_text", None) or ""
            hits.append(
                RawVectorHit(
                    document_id=int(row.document_id),
                    version_id=str(row.version_id),
                    chunk_index=int(row.chunk_index),
                    score=1.0 / (1.0 + distance),
                    payload={"chunk_text": chunk_text} if chunk_text else {},
                )
            )

        logger.info(
            "pgvector_backend.search tenant_id=%s hits=%d limit=%d",
            tenant_id, len(hits), limit,
        )
        return hits


# ── retriever protocol ────────────────────────────────────────────────

@runtime_checkable
class VectorRetriever(Protocol):
    """Retriever interface for vector similarity search."""

    async def search(
        self,
        *,
        tenant_id: str,
        query_embedding: list[float],
        limit: int,
        filters: VectorFilter,
    ) -> list[ScoredChunk]:
        """Search for chunks nearest to *query_embedding*."""
        ...


# ── default retriever (wraps backend + enrichment) ───────────────────

class DefaultVectorRetriever:
    """
    Production vector retriever.

    Wraps a ``VectorSearchBackend`` and converts ``RawVectorHit`` →
    ``ScoredChunk``.  Snippet enrichment from the document table
    is performed lazily via the session factory.
    """

    __slots__ = ("_backend", "_session_factory")

    def __init__(
        self,
        backend: VectorSearchBackend | None = None,
        session_factory=None,
    ) -> None:
        self._backend = backend or NullVectorSearchBackend()
        if session_factory is None:
            from app.db.session import AsyncSessionLocal
            session_factory = AsyncSessionLocal
        self._session_factory = session_factory

    async def search(
        self,
        *,
        tenant_id: str,
        query_embedding: list[float],
        limit: int,
        filters: VectorFilter,
    ) -> list[ScoredChunk]:
        raw_hits = await self._backend.search(
            query_vector=query_embedding,
            limit=limit,
            tenant_id=tenant_id,
            allowed_doc_ids=filters.document_ids,
        )

        if not raw_hits:
            return []

        # Check if any hits need snippet enrichment from DB
        needs_enrichment = any(
            not h.payload.get("chunk_text") for h in raw_hits
        )

        doc_map: dict[int, dict] = {}
        if needs_enrichment:
            doc_ids = {
                h.document_id for h in raw_hits
                if not h.payload.get("chunk_text")
            }
            doc_map = await self._load_document_meta(doc_ids, tenant_id)

        chunks: list[ScoredChunk] = []
        for h in raw_hits:
            # Phase 1: prefer persisted chunk_text from payload
            persisted_text = h.payload.get("chunk_text", "") or ""
            if persisted_text:
                snippet = persisted_text
            else:
                # Fallback for legacy data indexed before chunk_text persistence
                meta = doc_map.get(h.document_id, {})
                snippet = self._extract_snippet(
                    meta.get("content_text", ""), h.chunk_index,
                )

            meta = doc_map.get(h.document_id, {}) if doc_map else {}
            chunks.append(
                ScoredChunk(
                    chunk_id=make_chunk_id(h.document_id, h.chunk_index),
                    document_id=h.document_id,
                    version_id=h.version_id,
                    chunk_index=h.chunk_index,
                    score=h.score,
                    source="vector",
                    snippet=snippet,
                    title=meta.get("title") or h.payload.get("title"),
                )
            )

        logger.info(
            "vector_retriever.search tenant_id=%s hits=%d persisted=%d fallback=%d",
            tenant_id, len(chunks),
            sum(1 for h in raw_hits if h.payload.get("chunk_text")),
            sum(1 for h in raw_hits if not h.payload.get("chunk_text")),
        )
        return chunks

    async def _load_document_meta(
        self, doc_ids: set[int], tenant_id: str = "",
    ) -> dict[int, dict]:
        """Fetch title + content_text for enrichment.

        Phase 1 hotfix: when tenant_id is provided, adds defense-in-depth
        filtering so this query never leaks data across tenants.
        content_text is only loaded when needed for legacy snippet fallback.
        """
        from sqlalchemy import select
        from app.db.models.document import Document

        stmt = select(
            Document.id, Document.title, Document.content_text,
        ).where(Document.id.in_(doc_ids))

        if tenant_id:
            stmt = stmt.where(Document.tenant_id == tenant_id)

        async with self._session_factory() as db:
            result = await db.execute(stmt)
            rows = result.fetchall()

        return {
            row[0]: {"title": row[1], "content_text": row[2] or ""}
            for row in rows
        }

    @staticmethod
    def _extract_snippet(content_text: str, chunk_index: int) -> str:
        """
        Reconstruct the exact chunk from *content_text* using the same
        ``SemanticChunker`` as the ingest pipeline, then return its text.

        Falls back safely if chunking fails or chunk_index is out of range.
        """
        if not content_text:
            return ""

        try:
            from app.nlp import get_chunker

            chunker = get_chunker()
            chunks = chunker.chunk(content_text)

            if not chunks:
                # Chunker returned nothing — return a reasonable prefix
                return content_text[:1200]

            if chunk_index < len(chunks):
                return chunks[chunk_index].text
            else:
                # chunk_index out of range — use the last chunk
                logger.warning(
                    "snippet.chunk_index_out_of_range idx=%d total=%d",
                    chunk_index, len(chunks),
                )
                return chunks[-1].text
        except Exception:
            logger.warning("snippet.chunker_fallback chunk_index=%d", chunk_index)
            return content_text[:1200]


class NullVectorRetriever:
    """No-op retriever — always returns empty results."""

    __slots__ = ()

    async def search(self, **kw) -> list[ScoredChunk]:
        return []
