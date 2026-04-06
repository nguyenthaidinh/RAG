"""
Audit event repository (Phase 6.0).

Append-only — NO update / delete methods exist.
Supports dedupe via DB unique index on (tenant_id, event_type, dedupe_key, time_bucket).
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import ProgrammingError

from app.db.models.audit_event import AuditEvent

logger = logging.getLogger(__name__)


# ── DTOs ──────────────────────────────────────────────────────────────


class AuditEventCreate(BaseModel):
    """Input DTO for creating an audit event."""

    event_type: str
    tenant_id: str
    user_id: int | None = None
    actor: str
    severity: str
    ref_type: str | None = None
    ref_id: str | None = None
    metadata_json: dict[str, Any] = {}


class AuditEventFilters(BaseModel):
    """Filter parameters for listing audit events."""

    from_dt: datetime | None = None
    to_dt: datetime | None = None
    tenant_id: str | None = None
    user_id: int | None = None
    event_type: str | None = None
    severity: str | None = None


class DedupeArgs(BaseModel):
    """Arguments for deduplicated audit event creation."""

    dedupe_key: str
    time_bucket: str


# ── Repository ────────────────────────────────────────────────────────


class AuditEventRepository:
    """
    Append-only repository for audit events.

    Provides:
      create_event           — unconditional insert
      create_event_if_absent — deduplicated insert (uses DB unique index)
      list_events            — paginated listing with filters
      count_events           — count matching filters
    """

    __slots__ = ()

    async def create_event(
        self,
        db: AsyncSession,
        event: AuditEventCreate,
    ) -> AuditEvent:
        """Insert an audit event unconditionally."""
        row = AuditEvent(
            event_type=event.event_type,
            tenant_id=event.tenant_id,
            user_id=event.user_id,
            actor=event.actor,
            severity=event.severity,
            ref_type=event.ref_type,
            ref_id=event.ref_id,
            metadata_json=event.metadata_json,
        )
        db.add(row)
        await db.flush()
        await db.refresh(row)
        return row

    async def create_event_if_absent(
        self,
        db: AsyncSession,
        dedupe: DedupeArgs,
        event: AuditEventCreate,
    ) -> AuditEvent | None:
        """
        Insert audit event only if no matching dedupe row exists.

        Uses ON CONFLICT DO NOTHING against the partial unique index
        ``uq_audit_events_dedupe``.
        """
        metadata = dict(event.metadata_json)
        metadata["dedupe_key"] = dedupe.dedupe_key
        metadata["time_bucket"] = dedupe.time_bucket

        stmt = (
            pg_insert(AuditEvent)
            .values(
                event_type=event.event_type,
                tenant_id=event.tenant_id,
                user_id=event.user_id,
                actor=event.actor,
                severity=event.severity,
                ref_type=event.ref_type,
                ref_id=event.ref_id,
                metadata_json=metadata,
            )
            .on_conflict_do_nothing()
            .returning(AuditEvent)
        )
        result = await db.execute(stmt)
        row = result.scalars().first()
        if row is not None:
            await db.flush()
        return row

    async def list_events(
        self,
        db: AsyncSession,
        filters: AuditEventFilters,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditEvent], int]:
        """
        Return (items, total) for the given filters.

        Stable pagination: ORDER BY created_at DESC, id DESC.
        """
        try:
            base_q = select(AuditEvent)
            count_q = select(func.count(AuditEvent.id))

            base_q = self._apply_filters(base_q, filters)
            count_q = self._apply_filters(count_q, filters)

            total = (await db.execute(count_q)).scalar() or 0

            items_q = (
                base_q
                .order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await db.execute(items_q)
            items = list(result.scalars().all())

            return items, total

        except ProgrammingError:
            logger.warning(
                "audit_events table not found — returning empty audit list"
            )
            return [], 0

    async def count_events(
        self,
        db: AsyncSession,
        filters: AuditEventFilters,
    ) -> int:
        """Count events matching the given filters."""
        try:
            count_q = select(func.count(AuditEvent.id))
            count_q = self._apply_filters(count_q, filters)
            result = await db.execute(count_q)
            return result.scalar() or 0
        except ProgrammingError:
            return 0

    @staticmethod
    def _apply_filters(stmt, filters: AuditEventFilters):
        """Apply optional filters to a select statement."""
        if filters.from_dt is not None:
            stmt = stmt.where(AuditEvent.created_at >= filters.from_dt)
        if filters.to_dt is not None:
            stmt = stmt.where(AuditEvent.created_at <= filters.to_dt)
        if filters.tenant_id is not None:
            stmt = stmt.where(AuditEvent.tenant_id == filters.tenant_id)
        if filters.user_id is not None:
            stmt = stmt.where(AuditEvent.user_id == filters.user_id)
        if filters.event_type is not None:
            stmt = stmt.where(AuditEvent.event_type == filters.event_type)
        if filters.severity is not None:
            stmt = stmt.where(AuditEvent.severity == filters.severity)
        return stmt


# ── In-memory implementation for testing ──────────────────────────────


class InMemoryAuditEventRepository:
    """
    In-memory audit event repository for testing.

    Thread-unsafe — intended for single-process test runs.
    """

    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: list[AuditEvent] = []

    async def create_event(
        self,
        db: AsyncSession | None,
        event: AuditEventCreate,
    ) -> AuditEvent:
        import uuid
        from datetime import timezone

        row = AuditEvent(
            id=uuid.uuid4(),
            created_at=datetime.now(timezone.utc),
            event_type=event.event_type,
            tenant_id=event.tenant_id,
            user_id=event.user_id,
            actor=event.actor,
            severity=event.severity,
            ref_type=event.ref_type,
            ref_id=event.ref_id,
            metadata_json=dict(event.metadata_json),
        )
        self._store.append(row)
        return row

    async def create_event_if_absent(
        self,
        db: AsyncSession | None,
        dedupe: DedupeArgs,
        event: AuditEventCreate,
    ) -> AuditEvent | None:
        import uuid
        from datetime import timezone

        for existing in self._store:
            if (
                existing.tenant_id == event.tenant_id
                and existing.event_type == event.event_type
                and existing.metadata_json.get("dedupe_key") == dedupe.dedupe_key
                and existing.metadata_json.get("time_bucket") == dedupe.time_bucket
            ):
                return None

        metadata = dict(event.metadata_json)
        metadata["dedupe_key"] = dedupe.dedupe_key
        metadata["time_bucket"] = dedupe.time_bucket

        row = AuditEvent(
            id=uuid.uuid4(),
            created_at=datetime.now(timezone.utc),
            event_type=event.event_type,
            tenant_id=event.tenant_id,
            user_id=event.user_id,
            actor=event.actor,
            severity=event.severity,
            ref_type=event.ref_type,
            ref_id=event.ref_id,
            metadata_json=metadata,
        )
        self._store.append(row)
        return row

    async def list_events(
        self,
        db: AsyncSession | None,
        filters: AuditEventFilters,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[AuditEvent], int]:
        filtered = self._filter(filters)
        filtered.sort(key=lambda e: (e.created_at, str(e.id)), reverse=True)
        total = len(filtered)
        items = filtered[offset: offset + limit]
        return items, total

    async def count_events(
        self,
        db: AsyncSession | None,
        filters: AuditEventFilters,
    ) -> int:
        return len(self._filter(filters))

    def _filter(self, f: AuditEventFilters) -> list[AuditEvent]:
        result = list(self._store)
        if f.from_dt is not None:
            result = [e for e in result if e.created_at >= f.from_dt]
        if f.to_dt is not None:
            result = [e for e in result if e.created_at <= f.to_dt]
        if f.tenant_id is not None:
            result = [e for e in result if e.tenant_id == f.tenant_id]
        if f.user_id is not None:
            result = [e for e in result if e.user_id == f.user_id]
        if f.event_type is not None:
            result = [e for e in result if e.event_type == f.event_type]
        if f.severity is not None:
            result = [e for e in result if e.severity == f.severity]
        return result

    @property
    def count(self) -> int:
        return len(self._store)

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._store)
