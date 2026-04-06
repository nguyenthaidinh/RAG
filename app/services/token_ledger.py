"""
Token accounting service — billing-safe, additive, idempotent.

The ledger sits between the ingest pipeline and the billing system.
It guarantees:

1. **Additive**: every ``record_chunk_usage`` call adds records — never
   modifies or deletes existing ones.
2. **Idempotent**: re-ingesting the same document version
   (``tenant_id + document_id + version_id``) does NOT double-count.
3. **Queryable**: token totals can be summed per tenant, time window, and
   usage type for billing reports.

TODO (Phase 3.2): Wire to a DB-backed ChunkUsageRepository.
"""
from __future__ import annotations

import logging
from datetime import datetime

from app.nlp.types import Chunk
from app.services.metering import (
    ChunkUsage,
    ChunkUsageRepository,
    InMemoryChunkUsageRepository,
)

logger = logging.getLogger(__name__)


class TokenLedgerService:
    """
    Thin orchestration layer over ``ChunkUsageRepository``.

    All mutation is delegated to the repository; the service owns the
    idempotency guard and the mapping from ``Chunk`` → ``ChunkUsage``.
    """

    __slots__ = ("_repo",)

    def __init__(self, repo: ChunkUsageRepository) -> None:
        self._repo = repo

    # ── write ─────────────────────────────────────────────────────

    async def record_chunk_usage(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
        chunks: list[Chunk],
    ) -> list[ChunkUsage]:
        """
        Record one ``ChunkUsage`` per chunk.

        **Idempotent**: if records already exist for the
        ``(tenant_id, document_id, version_id)`` triple, the existing
        records are returned and nothing is written.

        Returns the (possibly pre-existing) usage records.
        """
        # Guard: idempotent per version
        if await self._repo.exists(tenant_id, document_id, version_id):
            existing = await self._repo.find_by_version(
                tenant_id, document_id, version_id,
            )
            logger.info(
                "metering.skipped tenant_id=%s doc_id=%s version_id=%s "
                "reason=already_recorded records=%d",
                tenant_id, document_id, version_id, len(existing),
            )
            return existing

        records = [
            ChunkUsage(
                tenant_id=tenant_id,
                document_id=document_id,
                version_id=version_id,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count,
                usage_type="ingest",
            )
            for chunk in chunks
        ]

        await self._repo.bulk_insert(records)

        total_tokens = sum(r.token_count for r in records)
        logger.info(
            "metering.recorded tenant_id=%s doc_id=%s version_id=%s "
            "chunks=%d tokens_total=%d",
            tenant_id, document_id, version_id, len(records), total_tokens,
        )
        return records

    # ── read ──────────────────────────────────────────────────────

    async def sum_tokens(
        self,
        *,
        tenant_id: str,
        usage_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int:
        """
        Return the total token count for *tenant_id*, optionally
        filtered by *usage_type* and/or time window.
        """
        return await self._repo.sum_tokens(
            tenant_id=tenant_id,
            usage_type=usage_type,
            since=since,
            until=until,
        )

    async def get_document_usage(
        self,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> list[ChunkUsage]:
        """Return all usage records for a specific document version."""
        return await self._repo.find_by_version(
            tenant_id, document_id, version_id,
        )


# ── module-level default (singleton) ─────────────────────────────────

_default_instance: TokenLedgerService | None = None


def get_token_ledger() -> TokenLedgerService:
    """
    Return the process-wide ``TokenLedgerService`` singleton.

    Uses ``InMemoryChunkUsageRepository`` in Phase 3.1.
    TODO (Phase 3.2): Replace with DB-backed repository.
    """
    global _default_instance
    if _default_instance is None:
        _default_instance = TokenLedgerService(InMemoryChunkUsageRepository())
    return _default_instance
