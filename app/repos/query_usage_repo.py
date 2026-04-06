"""
Query usage repository — idempotent persistence for query billing records.

Two implementations:

* **PgQueryUsageRepository** — PostgreSQL via async SQLAlchemy.
* **InMemoryQueryUsageRepository** — in-process dict for testing.

No raw query text passes through this layer.
"""
from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.query_usage import QueryUsage

logger = logging.getLogger(__name__)


class QueryUsageRepository(Protocol):
    """Persistence interface for query usage records."""

    async def get_by_idempotency(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
    ) -> QueryUsage | None:
        """Return existing record or ``None``."""
        ...
        
    async def insert_if_absent(
        self,
        *,
        usage: QueryUsage,
    ) -> tuple[QueryUsage, bool]:
        """
        Insert *usage* if no record with the same
        ``(tenant_id, idempotency_key)`` exists.

        Returns ``(record, was_inserted)`` — the returned record may be a
        pre-existing row when ``was_inserted is False``.
        """
        ...


class PgQueryUsageRepository:
    """
    PostgreSQL implementation of ``QueryUsageRepository``.

    Manages its own session via a ``session_factory`` — not coupled to
    the caller's transaction.
    """

    __slots__ = ("_session_factory",)

    def __init__(self, session_factory=None) -> None:
        if session_factory is None:
            from app.db.session import AsyncSessionLocal
            session_factory = AsyncSessionLocal
        self._session_factory = session_factory

    async def get_by_idempotency(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
    ) -> QueryUsage | None:
        async with self._session_factory() as db:
            result = await db.execute(
                select(QueryUsage).where(
                    QueryUsage.tenant_id == tenant_id,
                    QueryUsage.idempotency_key == idempotency_key,
                )
            )
            return result.scalars().first()

    async def insert_if_absent(
        self,
        *,
        usage: QueryUsage,
    ) -> tuple[QueryUsage, bool]:
        async with self._session_factory() as db:
            # Check for existing record
            result = await db.execute(
                select(QueryUsage).where(
                    QueryUsage.tenant_id == usage.tenant_id,
                    QueryUsage.idempotency_key == usage.idempotency_key,
                )
            )
            existing = result.scalars().first()
            if existing is not None:
                return existing, False

            db.add(usage)
            await db.commit()

            logger.debug(
                "query_usage_repo.inserted tenant_id=%s idempotency_key=%s",
                usage.tenant_id, usage.idempotency_key,
            )
            return usage, True


class InMemoryQueryUsageRepository:
    """
    In-memory implementation for testing.

    Keyed by ``(tenant_id, idempotency_key)``.  Not thread-safe —
    intended for single-process test runs only.
    """

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], QueryUsage] = {}

    async def get_by_idempotency(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
    ) -> QueryUsage | None:
        return self._store.get((tenant_id, idempotency_key))

    async def insert_if_absent(
        self,
        *,
        usage: QueryUsage,
    ) -> tuple[QueryUsage, bool]:
        key = (usage.tenant_id, usage.idempotency_key)
        if key in self._store:
            return self._store[key], False
        self._store[key] = usage
        return usage, True

    @property
    def count(self) -> int:
        """Number of stored records (test helper)."""
        return len(self._store)
