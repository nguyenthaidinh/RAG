"""Platform admin service (Phase 4 + Phase 7).

Orchestrates DB-backed source management, sync execution,
scheduling, and health tracking.

Responsibilities:
  - CRUD for onboarded sources
  - DB row → ``OnboardedSourceConfig`` mapping
  - Sync trigger with ``SourceSyncRun`` tracking
  - Sync coordinator locking (Phase 7)
  - Schedule management (Phase 7)
  - Health status reporting (Phase 7)
  - Auth secret masking on output

Does NOT:
  - Own the sync pipeline (delegates to ``sync_onboarded_source()``)
  - Run the scheduler loop (see ``source_sync_scheduler``)
  - Hard-code web-specific logic
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.onboarded_source import OnboardedSource
from app.db.models.source_sync_run import SourceSyncRun
from app.repos.source_platform_repo import SourcePlatformRepo
from app.schemas.source_platform import (
    SourceHealthResponse,
    SourceResponse,
    SyncRunResponse,
    mask_auth_config,
)
from app.services.source_platform.source_sync_coordinator import (
    get_sync_coordinator,
)
from app.services.source_platform.runtime_mapping import (
    row_to_runtime_config,
)

logger = logging.getLogger(__name__)

_MAX_ERROR_MESSAGE_LEN = 2000


class PlatformAdminService:
    """Admin service for managing onboarded sources and sync runs."""

    def __init__(self, repo: SourcePlatformRepo | None = None) -> None:
        self._repo = repo or SourcePlatformRepo()

    # ── Source CRUD ──────────────────────────────────────────────

    async def create_source(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        data: dict[str, Any],
    ) -> SourceResponse:
        """Create a new onboarded source."""

        # Check uniqueness
        existing = await self._repo.get_source_by_key(
            db, tenant_id=tenant_id, source_key=data["source_key"]
        )
        if existing:
            raise ValueError(
                f"Source with source_key='{data['source_key']}' "
                f"already exists for this tenant"
            )

        source = OnboardedSource(
            tenant_id=tenant_id,
            source_key=data["source_key"],
            name=data["name"],
            description=data.get("description"),
            connector_type=data["connector_type"],
            base_url=data["base_url"],
            auth_type=data.get("auth_type", "bearer"),
            auth_config=data.get("auth_config"),
            list_path=data.get("list_path", "/api/internal/knowledge/items"),
            detail_path_template=data.get("detail_path_template"),
            request_config=data.get("request_config"),
            mapping_config=data.get("mapping_config"),
            default_metadata=data.get("default_metadata"),
            is_active=data.get("is_active", True),
            sync_enabled=data.get("sync_enabled", False),
            sync_interval_minutes=data.get("sync_interval_minutes", 60),
        )

        self._repo.add_source(db, source)
        await db.flush()

        logger.info(
            "source_platform.admin.created source_id=%d source_key=%s "
            "tenant=%s connector=%s",
            source.id,
            source.source_key,
            tenant_id,
            source.connector_type,
        )

        return self._to_response(source)

    async def update_source(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        data: dict[str, Any],
    ) -> SourceResponse | None:
        """Update an existing source.  Returns None if not found.

        Uses ``exclude_unset=True`` semantics from the API layer:
        - field absent from ``data`` → keep current value
        - field present with ``None`` → clear to NULL (for nullable fields)
        - field present with value → set new value
        """

        source = await self._repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        # Fields that can be set to None (nullable in DB)
        _nullable = frozenset({
            "description", "auth_config", "detail_path_template",
            "request_config", "mapping_config", "default_metadata",
        })

        # All updatable fields
        updatable = (
            "name", "description", "base_url", "auth_type", "auth_config",
            "list_path", "detail_path_template", "request_config",
            "mapping_config", "default_metadata", "is_active",
        )
        for field_name in updatable:
            if field_name not in data:
                continue  # not sent → keep current
            value = data[field_name]
            if value is None and field_name not in _nullable:
                continue  # non-nullable field, ignore None
            setattr(source, field_name, value)

        await db.flush()

        logger.info(
            "source_platform.admin.updated source_id=%d source_key=%s tenant=%s",
            source.id,
            source.source_key,
            tenant_id,
        )

        return self._to_response(source)

    async def get_source(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
    ) -> SourceResponse | None:
        """Get a single source detail."""
        source = await self._repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None
        return self._to_response(source)

    async def list_sources(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        connector_type: str | None = None,
        is_active: bool | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[SourceResponse], int]:
        """List sources for a tenant."""
        sources, total = await self._repo.list_sources(
            db,
            tenant_id=tenant_id,
            connector_type=connector_type,
            is_active=is_active,
            page=page,
            page_size=page_size,
        )
        items = [self._to_response(s) for s in sources]
        return items, total

    # ── Sync trigger ─────────────────────────────────────────────

    async def trigger_sync(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        triggered_by: str = "api",
    ) -> dict[str, Any] | None:
        """Trigger a sync for one source.

        Returns sync summary dict, or ``None`` if source not found.
        """
        source = await self._repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        if not source.is_active:
            raise ValueError(
                f"Source '{source.source_key}' is inactive and cannot be synced. "
                f"Activate it first via PUT /sources/{source.id}"
            )

        # 1. Create sync run
        run = SourceSyncRun(
            tenant_id=tenant_id,
            source_id=source.id,
            source_key=source.source_key,
            status="running",
            triggered_by=triggered_by,
        )
        self._repo.add_sync_run(db, run)
        await db.flush()

        logger.info(
            "source_platform.admin.sync_start run_id=%d source_key=%s tenant=%s",
            run.id,
            source.source_key,
            tenant_id,
        )

        # 2. Map DB row → runtime config
        config = self._row_to_config(source)

        # 3. Run sync
        try:
            from app.services.source_platform.source_onboarding_service import (
                sync_onboarded_source,
            )

            result = await sync_onboarded_source(
                db,
                config=config,
                tenant_id=tenant_id,
                onboarded_source_id=source.id,
            )

            # 4. Update run with results
            run.status = "success"
            run.items_fetched = result.fetched_count
            run.items_upserted = result.upserted_count
            run.items_failed = result.failed_count
            # Phase 8: granular delta metrics
            run.items_created = result.created_count
            run.items_updated = result.updated_count
            run.items_unchanged = result.unchanged_count
            run.items_missing = result.missing_count
            run.items_reactivated = result.reactivated_count
            run.finished_at = datetime.now(timezone.utc)

            if result.errors:
                # Store first few errors as summary
                import json
                error_summary = json.dumps(
                    result.errors[:10], ensure_ascii=False
                )
                run.error_message = error_summary[:_MAX_ERROR_MESSAGE_LEN]

            # 5. Update source last_synced_at + Phase 7 health
            source.last_synced_at = run.finished_at
            await self._repo.update_health_on_success(
                db, source, now=run.finished_at
            )

            await db.flush()

            logger.info(
                "source_platform.admin.sync_done run_id=%d source_key=%s "
                "discovered=%d created=%d updated=%d unchanged=%d "
                "reactivated=%d missing=%d errors=%d",
                run.id,
                source.source_key,
                result.discovered_count,
                result.created_count,
                result.updated_count,
                result.unchanged_count,
                result.reactivated_count,
                result.missing_count,
                result.error_count,
            )

            return {
                "sync_run_id": run.id,
                "source_id": source.id,
                "source_key": source.source_key,
                "status": "success",
                "items_fetched": result.fetched_count,
                "items_created": result.created_count,
                "items_updated": result.updated_count,
                "items_unchanged": result.unchanged_count,
                "items_missing": result.missing_count,
                "items_reactivated": result.reactivated_count,
                "items_upserted": result.upserted_count,
                "items_failed": result.failed_count,
                "message": (
                    f"Synced: {result.created_count} created, "
                    f"{result.updated_count} updated, "
                    f"{result.unchanged_count} unchanged, "
                    f"{result.reactivated_count} reactivated, "
                    f"{result.missing_count} missing, "
                    f"{result.error_count} errors"
                ),
            }

        except Exception as exc:
            run.status = "failed"
            run.finished_at = datetime.now(timezone.utc)
            run.error_message = str(exc)[:_MAX_ERROR_MESSAGE_LEN]

            # Do NOT update source.last_synced_at on failure.
            # last_synced_at represents the most recent SUCCESSFUL sync.

            # Phase 7: track failure in health fields
            await self._repo.update_health_on_failure(
                db, source,
                error_message=str(exc)[:_MAX_ERROR_MESSAGE_LEN],
                now=run.finished_at,
            )

            await db.flush()

            logger.exception(
                "source_platform.admin.sync_failed run_id=%d source_key=%s",
                run.id,
                source.source_key,
            )

            return {
                "sync_run_id": run.id,
                "source_id": source.id,
                "source_key": source.source_key,
                "status": "failed",
                "items_fetched": run.items_fetched,
                "items_upserted": run.items_upserted,
                "items_failed": run.items_failed,
                "message": f"Sync failed: {str(exc)[:200]}",
            }

    # ── Sync run history ─────────────────────────────────────────

    async def list_sync_runs(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        limit: int = 20,
    ) -> tuple[list[SyncRunResponse], int] | None:
        """List sync runs for a source.  Returns None if source not found."""

        source = await self._repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        runs, total = await self._repo.list_sync_runs(
            db,
            tenant_id=tenant_id,
            source_id=source_id,
            limit=limit,
        )

        items = [
            SyncRunResponse(
                id=r.id,
                tenant_id=r.tenant_id,
                source_id=r.source_id,
                source_key=r.source_key,
                status=r.status,
                started_at=r.started_at,
                finished_at=r.finished_at,
                items_fetched=r.items_fetched,
                items_created=r.items_created,
                items_updated=r.items_updated,
                items_unchanged=r.items_unchanged,
                items_missing=r.items_missing,
                items_reactivated=r.items_reactivated,
                items_upserted=r.items_upserted,
                items_failed=r.items_failed,
                error_message=r.error_message,
                triggered_by=r.triggered_by,
                created_at=r.created_at,
            )
            for r in runs
        ]
        return items, total

    # ── Phase 7: Coordinated sync ────────────────────────────────

    async def trigger_sync_coordinated(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        triggered_by: str = "api",
    ) -> dict[str, Any] | None:
        """Trigger a sync with coordinator lock guard.

        Raises ``ValueError`` if the source is already syncing.
        Returns ``None`` if source not found.
        """
        coordinator = get_sync_coordinator()
        acquired = await coordinator.acquire(source_id)
        if not acquired:
            raise ValueError(
                f"Source {source_id} already has a sync in progress"
            )
        try:
            return await self.trigger_sync(
                db,
                tenant_id=tenant_id,
                source_id=source_id,
                triggered_by=triggered_by,
            )
        finally:
            await coordinator.release(source_id)

    # ── Phase 7: Schedule management ─────────────────────────────

    async def update_schedule(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
        data: dict[str, Any],
    ) -> SourceResponse | None:
        """Update sync schedule fields for a source."""
        source = await self._repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        if "sync_enabled" in data:
            source.sync_enabled = data["sync_enabled"]
        if "sync_interval_minutes" in data:
            source.sync_interval_minutes = data["sync_interval_minutes"]

        # If enabling and no next_sync_at set, compute initial schedule
        if source.sync_enabled and source.next_sync_at is None:
            from datetime import timedelta
            source.next_sync_at = (
                datetime.now(timezone.utc)
                + timedelta(minutes=source.sync_interval_minutes)
            )

        await db.flush()

        logger.info(
            "source_platform.admin.schedule_updated source_id=%d "
            "sync_enabled=%s interval=%d tenant=%s",
            source.id,
            source.sync_enabled,
            source.sync_interval_minutes,
            tenant_id,
        )

        return self._to_response(source)

    async def pause_source(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
    ) -> SourceResponse | None:
        """Pause scheduled sync (sets sync_enabled=False)."""
        source = await self._repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        source.sync_enabled = False
        await db.flush()

        logger.info(
            "source_platform.admin.paused source_id=%d tenant=%s",
            source.id,
            tenant_id,
        )
        return self._to_response(source)

    async def resume_source(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
    ) -> SourceResponse | None:
        """Resume scheduled sync (sets sync_enabled=True)."""
        source = await self._repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        source.sync_enabled = True

        # Compute next_sync_at if not already set
        if source.next_sync_at is None:
            from datetime import timedelta
            source.next_sync_at = (
                datetime.now(timezone.utc)
                + timedelta(minutes=source.sync_interval_minutes)
            )

        await db.flush()

        logger.info(
            "source_platform.admin.resumed source_id=%d tenant=%s",
            source.id,
            tenant_id,
        )
        return self._to_response(source)

    # ── Phase 7: Health ──────────────────────────────────────────

    async def get_health(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source_id: int,
    ) -> SourceHealthResponse | None:
        """Get health status for a source."""
        source = await self._repo.get_source(
            db, tenant_id=tenant_id, source_id=source_id
        )
        if not source:
            return None

        return SourceHealthResponse(
            source_id=source.id,
            source_key=source.source_key,
            is_active=source.is_active,
            sync_enabled=source.sync_enabled,
            health_status=self._compute_health_status(source),
            consecutive_failures=source.consecutive_failures or 0,
            last_success_at=source.last_success_at,
            last_failure_at=source.last_failure_at,
            last_sync_attempt_at=source.last_sync_attempt_at,
            last_error_message=source.last_error_message,
            next_sync_at=source.next_sync_at,
            sync_interval_minutes=source.sync_interval_minutes,
        )

    @staticmethod
    def _compute_health_status(source: OnboardedSource) -> str:
        """Derive health status string from source state."""
        if not source.is_active:
            return "inactive"
        if not source.sync_enabled:
            return "paused"
        failures = source.consecutive_failures or 0
        if failures == 0:
            return "healthy"
        if failures < 3:
            return "degraded"
        return "unhealthy"

    # ── Mapping helpers ──────────────────────────────────────────

    @staticmethod
    def _row_to_config(row: OnboardedSource) -> OnboardedSourceConfig:
        """Map a DB row to a runtime ``OnboardedSourceConfig``.

        Delegates to the shared ``row_to_runtime_config()`` helper
        to ensure all source platform services use identical mapping.
        """
        return row_to_runtime_config(row)

    @staticmethod
    def _to_response(source: OnboardedSource) -> SourceResponse:
        """Convert DB row to response schema with masked auth."""
        return SourceResponse(
            id=source.id,
            tenant_id=source.tenant_id,
            source_key=source.source_key,
            name=source.name,
            description=source.description,
            connector_type=source.connector_type,
            base_url=source.base_url,
            auth_type=source.auth_type,
            auth_config=mask_auth_config(source.auth_config),
            list_path=source.list_path,
            detail_path_template=source.detail_path_template,
            request_config=source.request_config,
            mapping_config=source.mapping_config,
            default_metadata=source.default_metadata,
            is_active=source.is_active,
            last_synced_at=source.last_synced_at,
            # Phase 7 fields
            sync_enabled=source.sync_enabled,
            sync_interval_minutes=source.sync_interval_minutes,
            next_sync_at=source.next_sync_at,
            last_sync_attempt_at=source.last_sync_attempt_at,
            health_status=PlatformAdminService._compute_health_status(source),
            created_at=source.created_at,
            updated_at=source.updated_at,
        )
