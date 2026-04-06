"""
Source document link repository (Phase 8 — Delta-Aware Sync).

Thin async data-access layer for source_document_links.
Follows ``SourcePlatformRepo`` patterns.  All queries tenant-scoped.
"""
from __future__ import annotations

import logging
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy import func as sa_func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.source_document_link import SourceDocumentLink

logger = logging.getLogger(__name__)


class SourceDocumentLinkRepo:
    """Async repository for source ↔ document link tracking."""

    # ── Single-item lookups ──────────────────────────────────────

    async def get_by_external_id(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        onboarded_source_id: int,
        external_id: str,
    ) -> SourceDocumentLink | None:
        """Find a link by its unique key (tenant + source + external_id)."""
        result = await db.execute(
            select(SourceDocumentLink).where(
                SourceDocumentLink.tenant_id == tenant_id,
                SourceDocumentLink.onboarded_source_id == onboarded_source_id,
                SourceDocumentLink.external_id == external_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_link_by_id(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        onboarded_source_id: int,
        link_id: int,
    ) -> SourceDocumentLink | None:
        """Find a link by its primary key, scoped to tenant + source."""
        result = await db.execute(
            select(SourceDocumentLink).where(
                SourceDocumentLink.id == link_id,
                SourceDocumentLink.tenant_id == tenant_id,
                SourceDocumentLink.onboarded_source_id == onboarded_source_id,
            )
        )
        return result.scalar_one_or_none()

    # ── Listing ──────────────────────────────────────────────────

    async def get_links_for_source(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        onboarded_source_id: int,
        status: str | None = None,
    ) -> list[SourceDocumentLink]:
        """List all links for a source, optionally filtered by status."""
        query = select(SourceDocumentLink).where(
            SourceDocumentLink.tenant_id == tenant_id,
            SourceDocumentLink.onboarded_source_id == onboarded_source_id,
        )
        if status is not None:
            query = query.where(SourceDocumentLink.status == status)
        result = await db.execute(query)
        return list(result.scalars().all())

    # ── Create / update ──────────────────────────────────────────

    @staticmethod
    def add(db: AsyncSession, link: SourceDocumentLink) -> None:
        """Add a new link to the session."""
        db.add(link)

    async def touch_seen(
        self,
        db: AsyncSession,
        link: SourceDocumentLink,
        *,
        now: datetime,
    ) -> None:
        """Update last_seen_at only (item appeared, content unchanged)."""
        link.last_seen_at = now
        await db.flush()

    async def touch_synced(
        self,
        db: AsyncSession,
        link: SourceDocumentLink,
        *,
        now: datetime,
        content_checksum: str | None,
        document_id: int | None,
        document_version_id: str | None = None,
        external_uri: str | None = None,
        remote_updated_at: datetime | None = None,
    ) -> None:
        """Update link after successful content sync."""
        link.last_seen_at = now
        link.last_synced_at = now
        link.content_checksum = content_checksum
        link.status = "active"
        if document_id is not None:
            link.document_id = document_id
        if document_version_id is not None:
            link.document_version_id = document_version_id
        if external_uri is not None:
            link.external_uri = external_uri
        if remote_updated_at is not None:
            link.remote_updated_at = remote_updated_at
        await db.flush()

    # ── Reactivation ─────────────────────────────────────────────

    async def reactivate(
        self,
        db: AsyncSession,
        link: SourceDocumentLink,
        *,
        now: datetime,
    ) -> None:
        """Set status back to 'active' for a previously missing link."""
        link.status = "active"
        link.last_seen_at = now
        await db.flush()

    # ── Missing sweep ────────────────────────────────────────────

    async def mark_missing(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        onboarded_source_id: int,
        seen_external_ids: set[str],
        now: datetime,
    ) -> int:
        """Mark all active links NOT in seen_external_ids as 'missing'.

        Returns the number of links marked missing.

        This is the fail-safe missing policy:
          - Only marks the LINK as missing.
          - Does NOT delete or modify any documents.
          - Items can be reactivated if they reappear.
        """
        if not seen_external_ids:
            # If nothing was seen, do NOT mark everything missing.
            # This protects against connector failures returning empty lists.
            return 0

        # Use a single UPDATE for efficiency
        stmt = (
            update(SourceDocumentLink)
            .where(
                SourceDocumentLink.tenant_id == tenant_id,
                SourceDocumentLink.onboarded_source_id == onboarded_source_id,
                SourceDocumentLink.status == "active",
                SourceDocumentLink.external_id.notin_(seen_external_ids),
            )
            .values(status="missing", updated_at=now)
            .execution_options(synchronize_session=False)
        )

        result = await db.execute(stmt)
        count = result.rowcount
        await db.flush()

        if count > 0:
            logger.info(
                "source_doc_links.marked_missing "
                "tenant_id=%s source_id=%d count=%d",
                tenant_id,
                onboarded_source_id,
                count,
            )

        return count

    async def mark_all_missing(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        onboarded_source_id: int,
        now: datetime,
    ) -> int:
        """Mark ALL active links for a source as 'missing'.

        Used when ``fetch_item_refs()`` returned a valid empty list,
        confirming the source has zero items.  All previously active
        links are now stale.

        Returns the number of links marked missing.
        """
        stmt = (
            update(SourceDocumentLink)
            .where(
                SourceDocumentLink.tenant_id == tenant_id,
                SourceDocumentLink.onboarded_source_id == onboarded_source_id,
                SourceDocumentLink.status == "active",
            )
            .values(status="missing", updated_at=now)
            .execution_options(synchronize_session=False)
        )

        result = await db.execute(stmt)
        count = result.rowcount
        await db.flush()

        if count > 0:
            logger.info(
                "source_doc_links.marked_all_missing "
                "tenant_id=%s source_id=%d count=%d",
                tenant_id,
                onboarded_source_id,
                count,
            )

        return count

    # ── Aggregation (Phase 1 Observability) ──────────────────────

    async def count_by_status(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        onboarded_source_id: int,
    ) -> dict[str, int]:
        """Count links grouped by status for a source.

        Returns a dict like ``{"active": 120, "missing": 3, "error": 1}``.
        Statuses with zero count are omitted.
        """
        query = (
            select(
                SourceDocumentLink.status,
                sa_func.count().label("cnt"),
            )
            .where(
                SourceDocumentLink.tenant_id == tenant_id,
                SourceDocumentLink.onboarded_source_id == onboarded_source_id,
            )
            .group_by(SourceDocumentLink.status)
        )
        result = await db.execute(query)
        return {row.status: row.cnt for row in result.all()}

    async def list_links_paginated(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        onboarded_source_id: int,
        status: str | None = None,
        q: str | None = None,
        document_id: int | None = None,
        has_document: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "last_seen_at",
        sort_order: str = "desc",
    ) -> tuple[list[SourceDocumentLink], int]:
        """Paginated, filtered listing of links for a source.

        Supports filtering by status, text search on external_id,
        specific document_id, and has_document boolean.
        """
        base = select(SourceDocumentLink).where(
            SourceDocumentLink.tenant_id == tenant_id,
            SourceDocumentLink.onboarded_source_id == onboarded_source_id,
        )
        count_q = (
            select(sa_func.count())
            .select_from(SourceDocumentLink)
            .where(
                SourceDocumentLink.tenant_id == tenant_id,
                SourceDocumentLink.onboarded_source_id == onboarded_source_id,
            )
        )

        # Apply filters
        if status is not None:
            base = base.where(SourceDocumentLink.status == status)
            count_q = count_q.where(SourceDocumentLink.status == status)

        if q is not None:
            pattern = f"%{q}%"
            base = base.where(SourceDocumentLink.external_id.ilike(pattern))
            count_q = count_q.where(
                SourceDocumentLink.external_id.ilike(pattern)
            )

        if document_id is not None:
            base = base.where(SourceDocumentLink.document_id == document_id)
            count_q = count_q.where(
                SourceDocumentLink.document_id == document_id
            )

        if has_document is True:
            base = base.where(SourceDocumentLink.document_id.isnot(None))
            count_q = count_q.where(
                SourceDocumentLink.document_id.isnot(None)
            )
        elif has_document is False:
            base = base.where(SourceDocumentLink.document_id.is_(None))
            count_q = count_q.where(SourceDocumentLink.document_id.is_(None))

        total = (await db.execute(count_q)).scalar() or 0

        # Sorting — allow safe column names only
        _ALLOWED_SORT = {
            "last_seen_at", "last_synced_at", "created_at",
            "updated_at", "external_id", "status",
        }
        col_name = sort_by if sort_by in _ALLOWED_SORT else "last_seen_at"
        col = getattr(SourceDocumentLink, col_name)
        order = col.desc() if sort_order == "desc" else col.asc()

        offset = (page - 1) * page_size
        query = base.order_by(order).offset(offset).limit(page_size)
        result = await db.execute(query)
        items = list(result.scalars().all())

        return items, total

