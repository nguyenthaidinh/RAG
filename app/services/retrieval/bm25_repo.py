"""
Repository layer for PostgreSQL full-text search (BM25-style ranking).

Isolates all FTS SQL behind a protocol so that the ``BM25Retriever``
never issues queries directly.  Uses ``plainto_tsquery`` for safe
parsing of user input (no syntax errors from malformed queries).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings

logger = logging.getLogger(__name__)

_ALLOWED_FTS_CONFIGS = {"simple", "english"}


def resolve_postgres_fts_config(raw_config: str | None = None) -> str:
    """
    Return a safe PostgreSQL FTS config name.

    Only literal configs in this whitelist are allowed because the value is
    inserted into an expression-index-friendly SQL literal below.
    """
    config = (raw_config or getattr(settings, "POSTGRES_FTS_CONFIG", "simple")).lower().strip()
    if config not in _ALLOWED_FTS_CONFIGS:
        logger.warning(
            "bm25_repo.invalid_fts_config config=%s falling_back=simple",
            config,
        )
        return "simple"
    return config


@dataclass(frozen=True)
class BM25Hit:
    """Raw FTS result from the repository layer."""

    document_id: int
    tenant_id: str
    version_id: str      # checksum
    title: str | None
    rank: float           # ts_rank score
    headline: str         # ts_headline snippet


class BM25Repository(Protocol):
    """Persistence interface for BM25 full-text search."""

    async def search(
        self,
        *,
        tenant_id: str,
        query_text: str,
        limit: int,
        allowed_doc_ids: Sequence[int],
    ) -> list[BM25Hit]:
        """
        Execute a full-text search scoped to *tenant_id* and
        *allowed_doc_ids*.  Returns ranked results with headline snippets.
        """
        ...


class PgBM25Repository:
    """
    PostgreSQL FTS implementation using ``ts_rank`` + ``ts_headline``.

    Searches ``documents.content_text`` via a GIN FTS index. The text search
    config is intentionally limited to a small whitelist.

    Filtering:
      - ``tenant_id`` — hard tenant isolation
      - ``id = ANY(:doc_ids)`` — access-policy enforcement
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
        tenant_id: str,
        query_text: str,
        limit: int,
        allowed_doc_ids: Sequence[int],
    ) -> list[BM25Hit]:
        if not allowed_doc_ids or not query_text.strip():
            return []

        fts_config = resolve_postgres_fts_config()

        # Safe literal interpolation: fts_config is whitelisted to
        # {"simple", "english"} before it reaches this SQL string.
        stmt = text(f"""
            SELECT d.id,
                   d.tenant_id,
                   d.checksum,
                   d.title,
                   ts_rank(
                       to_tsvector('{fts_config}', COALESCE(d.content_text, '')),
                       plainto_tsquery('{fts_config}', :query)
                   ) AS rank,
                   ts_headline(
                       '{fts_config}',
                       COALESCE(d.content_text, ''),
                       plainto_tsquery('{fts_config}', :query),
                       'MaxFragments=3,MaxWords=50,MinWords=15'
                   ) AS headline
              FROM documents d
             WHERE to_tsvector('{fts_config}', COALESCE(d.content_text, ''))
                   @@ plainto_tsquery('{fts_config}', :query)
               AND d.tenant_id = :tenant_id
               AND d.id = ANY(:doc_ids)
             ORDER BY rank DESC
             LIMIT :lim
        """)

        async with self._session_factory() as db:
            result = await db.execute(
                stmt,
                {
                    "query": query_text,
                    "tenant_id": tenant_id,
                    "doc_ids": list(allowed_doc_ids),
                    "lim": limit,
                },
            )
            rows = result.fetchall()

        hits: list[BM25Hit] = []
        for row in rows:
            hits.append(
                BM25Hit(
                    document_id=row[0],
                    tenant_id=row[1],
                    version_id=row[2],
                    title=row[3],
                    rank=float(row[4]),
                    headline=row[5],
                )
            )

        logger.info(
            "bm25_repo.search tenant_id=%s hits=%d limit=%d fts_config=%s",
            tenant_id, len(hits), limit, fts_config,
        )
        return hits


class NullBM25Repository:
    """No-op BM25 repository — returns empty results."""

    async def search(self, **kw) -> list[BM25Hit]:
        return []
