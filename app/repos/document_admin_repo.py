"""
Document admin repository (Phase 2A).

Read-only data access layer for document administration.
Separated from DocumentRepo (write/ingest) by design.

Rules:
  - All queries MUST scope by tenant_id (defense-in-depth)
  - No raw text logging
  - Read-only — no INSERT / UPDATE / DELETE on documents
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import BigInteger, Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document

logger = logging.getLogger(__name__)


@dataclass
class DocumentListFilters:
    """Filters for document listing."""

    tenant_id: str | None = None
    q: str | None = None
    status: str | None = None
    representation_type: str | None = None
    source: str | None = None


class DocumentAdminRepo:
    """Read-only repository for document administration."""

    async def list_documents(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        filters: DocumentListFilters | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Document], int]:
        """
        List documents scoped by tenant with optional filters.

        Returns (documents, total_count).
        """
        base = select(Document).where(Document.tenant_id == tenant_id)

        if filters:
            if filters.q:
                q_like = f"%{filters.q}%"
                base = base.where(
                    or_(
                        Document.title.ilike(q_like),
                        Document.external_id.ilike(q_like),
                        Document.source.ilike(q_like),
                    )
                )
            if filters.status:
                base = base.where(Document.status == filters.status)
            if filters.representation_type:
                base = base.where(
                    Document.representation_type == filters.representation_type
                )
            if filters.source:
                base = base.where(Document.source == filters.source)

        # Count
        count_stmt = select(func.count()).select_from(base.subquery())
        total = (await db.execute(count_stmt)).scalar() or 0

        # Paginate
        offset = (page - 1) * page_size
        items_stmt = (
            base
            .order_by(Document.updated_at.desc(), Document.id.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await db.execute(items_stmt)
        items = list(result.scalars().all())

        return items, total

    async def get_document_by_id(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
    ) -> Document | None:
        """Get a single document scoped by tenant. Returns None if not found."""
        stmt = select(Document).where(
            Document.id == document_id,
            Document.tenant_id == tenant_id,
        )
        result = await db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_parent(
        self,
        db: AsyncSession,
        *,
        parent_id: int,
        tenant_id: str,
    ) -> Document | None:
        """Get parent document."""
        return await self.get_document_by_id(
            db, document_id=parent_id, tenant_id=tenant_id,
        )

    async def get_children(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
    ) -> list[Document]:
        """Get child documents (synthesized representations)."""
        stmt = (
            select(Document)
            .where(
                Document.parent_document_id == document_id,
                Document.tenant_id == tenant_id,
            )
            .order_by(Document.created_at.desc())
        )
        result = await db.execute(stmt)
        return list(result.scalars().all())

    async def get_related_representations(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
    ) -> dict:
        """
        Get the full family: parent + children.

        Returns {
            "parent": Document | None,
            "children": list[Document],
        }
        """
        doc = await self.get_document_by_id(
            db, document_id=document_id, tenant_id=tenant_id,
        )
        if not doc:
            return {"parent": None, "children": []}

        parent = None
        if doc.parent_document_id:
            parent = await self.get_parent(
                db,
                parent_id=doc.parent_document_id,
                tenant_id=tenant_id,
            )

        children = await self.get_children(
            db, document_id=document_id, tenant_id=tenant_id,
        )

        return {"parent": parent, "children": children}

    async def list_document_events(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
        limit: int = 50,
    ) -> list:
        """Document events removed in CTDT fork — returns empty list."""
        return []

    async def get_distinct_sources(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
    ) -> list[str]:
        """Get distinct source values for filter dropdowns."""
        stmt = (
            select(Document.source)
            .where(Document.tenant_id == tenant_id)
            .distinct()
            .order_by(Document.source)
        )
        result = await db.execute(stmt)
        return [row[0] for row in result.fetchall()]
