"""
Source platform repository (Phase 4 + Phase 7).

Thin async data-access layer following ``DocumentRepo`` patterns.
All queries are tenant-scoped.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.onboarded_source import OnboardedSource
from app.db.models.source_sync_run import SourceSyncRun


class SourcePlatformRepo:
    """Async repository for onboarded sources and sync runs."""

    # ── Source CRUD ──────────────────────────────────────────────

    async def get_source(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
    ) -> OnboardedSource | None:
        result = await db.execute(
            select(OnboardedSource).where(
                OnboardedSource.tenant_id == tenant_id,
                OnboardedSource.id == source_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_source_by_key(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_key: str,
    ) -> OnboardedSource | None:
        result = await db.execute(
            select(OnboardedSource).where(
                OnboardedSource.tenant_id == tenant_id,
                OnboardedSource.source_key == source_key,
            )
        )
        return result.scalar_one_or_none()

    async def list_sources(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        connector_type: str | None = None,
        is_active: bool | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[OnboardedSource], int]:
        base = select(OnboardedSource).where(
            OnboardedSource.tenant_id == tenant_id
        )
        count_q = select(func.count()).select_from(
            OnboardedSource
        ).where(OnboardedSource.tenant_id == tenant_id)

        if connector_type is not None:
            base = base.where(OnboardedSource.connector_type == connector_type)
            count_q = count_q.where(OnboardedSource.connector_type == connector_type)
        if is_active is not None:
            base = base.where(OnboardedSource.is_active == is_active)
            count_q = count_q.where(OnboardedSource.is_active == is_active)

        total = (await db.execute(count_q)).scalar() or 0

        offset = (page - 1) * page_size
        query = (
            base
            .order_by(OnboardedSource.created_at.desc())
            .offset(offset)
            .limit(page_size)
        )
        result = await db.execute(query)
        items = list(result.scalars().all())

        return items, total

    @staticmethod
    def add_source(db: AsyncSession, source: OnboardedSource) -> None:
        db.add(source)

    # ── Sync runs ────────────────────────────────────────────────

    @staticmethod
    def add_sync_run(db: AsyncSession, run: SourceSyncRun) -> None:
        db.add(run)

    async def list_sync_runs(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        limit: int = 20,
    ) -> tuple[list[SourceSyncRun], int]:
        base = select(SourceSyncRun).where(
            SourceSyncRun.tenant_id == tenant_id,
            SourceSyncRun.source_id == source_id,
        )
        count_q = select(func.count()).select_from(
            SourceSyncRun
        ).where(
            SourceSyncRun.tenant_id == tenant_id,
            SourceSyncRun.source_id == source_id,
        )

        total = (await db.execute(count_q)).scalar() or 0

        query = (
            base
            .order_by(SourceSyncRun.started_at.desc())
            .limit(limit)
        )
        result = await db.execute(query)
        items = list(result.scalars().all())

        return items, total

    async def get_sync_run(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        run_id: int,
    ) -> SourceSyncRun | None:
        """Get a single sync run by ID, scoped to tenant + source."""
        result = await db.execute(
            select(SourceSyncRun).where(
                SourceSyncRun.id == run_id,
                SourceSyncRun.tenant_id == tenant_id,
                SourceSyncRun.source_id == source_id,
            )
        )
        return result.scalar_one_or_none()


    # ── Phase 7: Scheduling & Health ─────────────────────────────

    async def get_due_sources(
        self,
        db: AsyncSession,
        *,
        now: datetime,
    ) -> list[OnboardedSource]:
        """Find sources that are enabled, active, and due for sync.

        A source is due when ``next_sync_at <= now`` or
        ``next_sync_at IS NULL`` (never scheduled yet).
        """
        query = select(OnboardedSource).where(
            OnboardedSource.sync_enabled.is_(True),
            OnboardedSource.is_active.is_(True),
            (
                (OnboardedSource.next_sync_at <= now)
                | (OnboardedSource.next_sync_at.is_(None))
            ),
        )
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def update_health_on_success(
        db: AsyncSession,
        source: OnboardedSource,
        *,
        now: datetime,
    ) -> None:
        """Record a successful sync in health fields."""
        source.last_sync_attempt_at = now
        source.last_success_at = now
        source.consecutive_failures = 0
        source.last_error_message = None
        source.next_sync_at = now + timedelta(
            minutes=source.sync_interval_minutes
        )
        await db.flush()

    @staticmethod
    async def update_health_on_failure(
        db: AsyncSession,
        source: OnboardedSource,
        *,
        error_message: str,
        now: datetime,
    ) -> None:
        """Record a failed sync in health fields."""
        source.last_sync_attempt_at = now
        source.last_failure_at = now
        source.consecutive_failures = (source.consecutive_failures or 0) + 1
        source.last_error_message = error_message[:2000]
        # Still advance next_sync_at so scheduler doesn't hammer a failing source
        source.next_sync_at = now + timedelta(
            minutes=source.sync_interval_minutes
        )
        await db.flush()

