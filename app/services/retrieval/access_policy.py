"""
Tenant-level access control for the retrieval engine.

The access layer runs FIRST — before any search or scoring.  Retrieval
components never decide access; they only receive a pre-filtered set of
document IDs.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.services.retrieval.types import QueryScope

logger = logging.getLogger(__name__)


@runtime_checkable
class AccessPolicy(Protocol):
    """Determines which documents a user may retrieve."""

    async def allowed_documents(
        self,
        *,
        tenant_id: str,
        user_id: int,
        scope: QueryScope,
    ) -> set[int]:
        """Return IDs of documents the user is permitted to query."""
        ...


class DefaultTenantAccessPolicy:
    """
    Default policy: a user may query ONLY documents owned by their
    tenant that are in a queryable state (``ready`` or ``indexed``).

    Uses its own DB session — no coupling to caller transaction.
    """

    __slots__ = ("_session_factory",)

    def __init__(self, session_factory=None) -> None:
        if session_factory is None:
            from app.db.session import AsyncSessionLocal
            session_factory = AsyncSessionLocal
        self._session_factory = session_factory

    async def allowed_documents(
        self,
        *,
        tenant_id: str,
        user_id: int,
        scope: QueryScope,
    ) -> set[int]:
        async with self._session_factory() as db:
            result = await db.execute(
                select(Document.id).where(
                    Document.tenant_id == tenant_id,
                    Document.status.in_(("ready", "indexed")),
                )
            )
            doc_ids = {row[0] for row in result.fetchall()}

        logger.info(
            "access_policy.resolved tenant_id=%s user_id=%d doc_count=%d",
            tenant_id, user_id, len(doc_ids),
        )
        return doc_ids
