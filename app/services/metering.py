"""
Usage metering for the ingest pipeline.

Each ``ChunkUsage`` record is an **immutable** accounting event emitted
once per chunk during document ingest.  Records are the raw material for
billing aggregation (handled by ``TokenLedgerService``).

Persistence is isolated behind ``ChunkUsageRepository`` so that the
in-memory implementation used in Phase 3.1 can be swapped for a
DB-backed one without changing any callers.

TODO (Phase 3.2): Replace InMemoryChunkUsageRepository with a
DB-backed implementation backed by a ``chunk_usage`` table.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

logger = logging.getLogger(__name__)


# ── immutable usage record ────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ChunkUsage:
    """One record per chunk — never mutated after creation."""

    tenant_id: str
    document_id: int
    version_id: str
    chunk_index: int
    token_count: int
    usage_type: str                # "ingest"
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# ── repository protocol ──────────────────────────────────────────────

class ChunkUsageRepository(Protocol):
    """
    Persistence interface for chunk usage records.

    Implementations MUST be safe under concurrent writes within a
    single process.  Cross-process safety is deferred to the DB-backed
    implementation.
    """

    async def bulk_insert(self, records: list[ChunkUsage]) -> None:
        """Persist a batch of records atomically."""
        ...

    async def exists(
        self,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> bool:
        """Return True if any record exists for the given key."""
        ...

    async def find_by_version(
        self,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> list[ChunkUsage]:
        """Return all records for a specific document version."""
        ...

    async def sum_tokens(
        self,
        *,
        tenant_id: str,
        usage_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int:
        """Aggregate token count for a tenant, optionally filtered."""
        ...


# ── in-memory implementation (Phase 3.1) ─────────────────────────────

class InMemoryChunkUsageRepository:
    """
    In-memory implementation of ``ChunkUsageRepository``.

    Suitable for development and testing.  Data does **not** survive
    process restarts.

    TODO (Phase 3.2): Replace with a DB-backed implementation.
    """

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: list[ChunkUsage] = []

    async def bulk_insert(self, records: list[ChunkUsage]) -> None:
        self._store.extend(records)

    async def exists(
        self,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> bool:
        return any(
            r.tenant_id == tenant_id
            and r.document_id == document_id
            and r.version_id == version_id
            for r in self._store
        )

    async def find_by_version(
        self,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> list[ChunkUsage]:
        return [
            r for r in self._store
            if r.tenant_id == tenant_id
            and r.document_id == document_id
            and r.version_id == version_id
        ]

    async def sum_tokens(
        self,
        *,
        tenant_id: str,
        usage_type: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> int:
        total = 0
        for r in self._store:
            if r.tenant_id != tenant_id:
                continue
            if usage_type is not None and r.usage_type != usage_type:
                continue
            if since is not None and r.created_at < since:
                continue
            if until is not None and r.created_at >= until:
                continue
            total += r.token_count
        return total
