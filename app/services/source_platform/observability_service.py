"""Source platform observability service (Phase 1 Observability).

Pure read-side service for operator-grade observability of onboarded
sources.  Composes data from repos into rich overview, link detail,
and sync run detail responses with derived attention flags and
operational status.

Does NOT:
  - Mutate any data
  - Trigger syncs or manage schedules
  - Own command/control logic (that stays in PlatformAdminService)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.onboarded_source import OnboardedSource
from app.db.models.source_document_link import SourceDocumentLink
from app.db.models.source_sync_run import SourceSyncRun
from app.repos.source_document_link_repo import SourceDocumentLinkRepo
from app.repos.source_platform_repo import SourcePlatformRepo
from app.schemas.source_platform import (
    SourceAttentionSummary,
    SourceLinkCounts,
    SourceLinkDetailResponse,
    SourceLinkItemResponse,
    SourceLinkListResponse,
    SourceOverviewResponse,
    SyncRunDetailResponse,
)

logger = logging.getLogger(__name__)

# Stale threshold multiplier: source is stale when no success for
# this many multiples of its sync_interval_minutes.
_STALE_MULTIPLIER = 3

# Consecutive failure threshold for "failing" status.
_FAILING_THRESHOLD = 3


class SourcePlatformObservabilityService:
    """Read-side observability service for source platform."""

    def __init__(
        self,
        source_repo: SourcePlatformRepo | None = None,
        link_repo: SourceDocumentLinkRepo | None = None,
    ) -> None:
        self._source_repo = source_repo or SourcePlatformRepo()
        self._link_repo = link_repo or SourceDocumentLinkRepo()

    # ── Source Overview ──────────────────────────────────────────

    async def get_source_overview(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
    ) -> SourceOverviewResponse | None:
        """Build a control-room overview for a single source."""
        source = await self._source_repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        # Aggregate link counts in a single GROUP BY query
        status_counts = await self._link_repo.count_by_status(
            db,
            tenant_id=tenant_id,
            onboarded_source_id=source.id,
        )

        link_counts = SourceLinkCounts(
            total=sum(status_counts.values()),
            active=status_counts.get("active", 0),
            missing=status_counts.get("missing", 0),
            error=status_counts.get("error", 0),
        )

        attention = self._derive_attention(source, link_counts)
        operational_status = self._derive_operational_status(
            source, attention, link_counts
        )

        return SourceOverviewResponse(
            source_id=source.id,
            tenant_id=source.tenant_id,
            source_key=source.source_key,
            name=source.name,
            description=source.description,
            connector_type=source.connector_type,
            is_active=source.is_active,
            sync_enabled=source.sync_enabled,
            sync_interval_minutes=source.sync_interval_minutes,
            next_sync_at=source.next_sync_at,
            last_synced_at=source.last_synced_at,
            last_success_at=source.last_success_at,
            last_failure_at=source.last_failure_at,
            consecutive_failures=source.consecutive_failures or 0,
            last_error_message=source.last_error_message,
            link_counts=link_counts,
            attention=attention,
            operational_status=operational_status,
            created_at=source.created_at,
            updated_at=source.updated_at,
        )

    # ── Source Links List ────────────────────────────────────────

    async def list_source_links(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        status: str | None = None,
        q: str | None = None,
        document_id: int | None = None,
        has_document: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "last_seen_at",
        sort_order: str = "desc",
    ) -> SourceLinkListResponse | None:
        """Paginated, filtered listing of source document links."""
        # Verify source exists
        source = await self._source_repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        links, total = await self._link_repo.list_links_paginated(
            db,
            tenant_id=tenant_id,
            onboarded_source_id=source.id,
            status=status,
            q=q,
            document_id=document_id,
            has_document=has_document,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        items = [self._link_to_item(lnk) for lnk in links]

        return SourceLinkListResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    # ── Source Link Detail ───────────────────────────────────────

    async def get_source_link_detail(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        link_id: int,
    ) -> SourceLinkDetailResponse | None:
        """Get detailed view of a single link with source context."""
        source = await self._source_repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        link = await self._link_repo.get_link_by_id(
            db,
            tenant_id=tenant_id,
            onboarded_source_id=source.id,
            link_id=link_id,
        )
        if not link:
            return None

        return SourceLinkDetailResponse(
            id=link.id,
            tenant_id=link.tenant_id,
            onboarded_source_id=link.onboarded_source_id,
            source_key=link.source_key,
            external_id=link.external_id,
            status=link.status,
            document_id=link.document_id,
            # Transitional fallback: prefer version_id, fall back to checksum
            document_version_id=link.document_version_id or link.content_checksum,
            content_checksum=link.content_checksum,
            external_uri=link.external_uri,
            remote_updated_at=link.remote_updated_at,
            last_seen_at=link.last_seen_at,
            last_synced_at=link.last_synced_at,
            created_at=link.created_at,
            updated_at=link.updated_at,
            metadata_json=link.metadata_json,
            has_document=link.document_id is not None,
            is_missing=link.status == "missing",
            source_name=source.name,
            source_connector_type=source.connector_type,
        )

    # ── Sync Run Detail ─────────────────────────────────────────

    async def get_sync_run_detail(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        run_id: int,
    ) -> SyncRunDetailResponse | None:
        """Get detailed view of a single sync run with derived outcome."""
        # Verify source exists (tenant-scoping)
        source = await self._source_repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        run = await self._source_repo.get_sync_run(
            db,
            tenant_id=tenant_id,
            source_id=source_id,
            run_id=run_id,
        )
        if not run:
            return None

        return SyncRunDetailResponse(
            id=run.id,
            tenant_id=run.tenant_id,
            source_id=run.source_id,
            source_key=run.source_key,
            status=run.status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            duration_seconds=self._compute_duration(run),
            items_fetched=run.items_fetched,
            items_created=run.items_created,
            items_updated=run.items_updated,
            items_unchanged=run.items_unchanged,
            items_missing=run.items_missing,
            items_reactivated=run.items_reactivated,
            items_upserted=run.items_upserted,
            items_failed=run.items_failed,
            error_message=run.error_message,
            triggered_by=run.triggered_by,
            created_at=run.created_at,
            derived_outcome=self._derive_run_outcome(run),
        )

    # ── Derivation helpers ───────────────────────────────────────

    @staticmethod
    def _derive_attention(
        source: OnboardedSource,
        link_counts: SourceLinkCounts,
    ) -> SourceAttentionSummary:
        """Derive attention flags from source health and link counts."""
        if not source.is_active or not source.sync_enabled:
            return SourceAttentionSummary()

        failures = source.consecutive_failures or 0

        is_failing = failures >= _FAILING_THRESHOLD
        is_degraded = (0 < failures < _FAILING_THRESHOLD) or (
            link_counts.error > 0
        )

        # Stale: sync is enabled, has succeeded before, but no success
        # for longer than _STALE_MULTIPLIER × sync_interval_minutes.
        is_stale = False
        if source.last_success_at is not None:
            now = datetime.now(timezone.utc)
            stale_minutes = _STALE_MULTIPLIER * source.sync_interval_minutes
            elapsed = (now - source.last_success_at).total_seconds() / 60
            is_stale = elapsed > stale_minutes

        needs_attention = is_stale or is_failing or is_degraded

        return SourceAttentionSummary(
            needs_attention=needs_attention,
            is_stale=is_stale,
            is_degraded=is_degraded,
            is_failing=is_failing,
        )

    @staticmethod
    def _derive_operational_status(
        source: OnboardedSource,
        attention: SourceAttentionSummary,
        link_counts: SourceLinkCounts,
    ) -> str:
        """Derive an operational status label for the source.

        Taxonomy (ordered by severity):
          - inactive: source is deactivated
          - paused:   sync scheduling is disabled
          - failing:  >= 3 consecutive sync failures
          - stale:    no successful sync for > 3× interval
          - degraded: some sync failures (< 3) or error links
          - warning:  no failures, but missing links detected
          - healthy:  everything nominal
        """
        if not source.is_active:
            return "inactive"
        if not source.sync_enabled:
            return "paused"
        if attention.is_failing:
            return "failing"
        if attention.is_stale:
            return "stale"
        if attention.is_degraded:
            return "degraded"
        if link_counts.missing > 0:
            return "warning"
        return "healthy"

    @staticmethod
    def _derive_run_outcome(run: SourceSyncRun) -> str:
        """Derive operator-friendly outcome from run status + counters.

        Possible values:
          - ``running``: sync still in progress
          - ``success``: completed with zero item failures
          - ``partial``: completed but some items failed
          - ``failed``:  sync itself failed
        """
        if run.status == "failed":
            return "failed"
        if run.status == "running":
            return "running"
        # status == "success"
        if run.items_failed > 0:
            return "partial"
        return "success"

    @staticmethod
    def _compute_duration(run: SourceSyncRun) -> float | None:
        """Compute duration in seconds from started_at to finished_at."""
        if run.started_at and run.finished_at:
            return (run.finished_at - run.started_at).total_seconds()
        return None

    @staticmethod
    def _link_to_item(link: SourceDocumentLink) -> SourceLinkItemResponse:
        """Convert a SourceDocumentLink ORM row to list item response."""
        return SourceLinkItemResponse(
            id=link.id,
            tenant_id=link.tenant_id,
            onboarded_source_id=link.onboarded_source_id,
            external_id=link.external_id,
            status=link.status,
            document_id=link.document_id,
            # Transitional fallback: prefer version_id, fall back to checksum
            document_version_id=link.document_version_id or link.content_checksum,
            content_checksum=link.content_checksum,
            external_uri=link.external_uri,
            remote_updated_at=link.remote_updated_at,
            last_seen_at=link.last_seen_at,
            last_synced_at=link.last_synced_at,
            created_at=link.created_at,
            updated_at=link.updated_at,
        )
