"""
Source sync scheduler (Phase 7 — Scheduled Sync).

Background polling loop that finds due sources and triggers syncs.
Follows the ``UsageLogDispatcher`` pattern for lifecycle management.

Lifecycle::

    # Startup (in main.py lifespan)
    scheduler = get_sync_scheduler()
    scheduler.start()

    # Shutdown
    await scheduler.stop()

Design constraints:
  - Single asyncio.Task polling loop
  - Each tick opens its own DB session (no shared session)
  - Per-source errors are caught and recorded to health fields
  - Coordinator lock prevents overlapping syncs on same source
  - Never crashes the app if one source fails
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from app.core.config import settings

logger = logging.getLogger(__name__)


class SourceSyncScheduler:
    """Background scheduler for source sync polling."""

    def __init__(
        self,
        *,
        tick_seconds: int | None = None,
    ) -> None:
        self._tick = tick_seconds or settings.SOURCE_SYNC_SCHEDULER_TICK_SECONDS
        self._task: asyncio.Task | None = None
        self._running = False

    # ── Public API ───────────────────────────────────────────────

    def start(self) -> None:
        """Start the polling loop.  Idempotent."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="source_sync_scheduler")
        logger.info(
            "source_sync_scheduler.started tick_seconds=%d",
            self._tick,
        )

    async def stop(self) -> None:
        """Stop the polling loop.  Idempotent."""
        if not self._running and self._task is None:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("source_sync_scheduler.stopped")

    # ── Polling loop ─────────────────────────────────────────────

    async def _loop(self) -> None:
        """Main polling loop — runs until stopped."""
        while self._running:
            try:
                await self._tick_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("source_sync_scheduler.tick_error")

            # Sleep with cancellation support
            try:
                await asyncio.sleep(self._tick)
            except asyncio.CancelledError:
                break

    async def _tick_once(self) -> None:
        """Single scheduler tick: find due sources and sync them."""
        from app.db.session import AsyncSessionLocal
        from app.repos.source_platform_repo import SourcePlatformRepo
        from app.services.source_platform.source_sync_coordinator import (
            get_sync_coordinator,
        )
        from app.services.source_platform.platform_admin_service import (
            PlatformAdminService,
        )

        now = datetime.now(timezone.utc)
        repo = SourcePlatformRepo()
        coordinator = get_sync_coordinator()
        svc = PlatformAdminService(repo=repo)

        # Short-lived session for the due-sources query only.
        # Source objects are detached after close but their scalar
        # attributes (id, tenant_id, source_key) remain readable
        # because expire_on_commit=False in session config.
        async with AsyncSessionLocal() as db:
            due_sources = await repo.get_due_sources(db, now=now)

        if not due_sources:
            return

        logger.info(
            "source_sync_scheduler.tick due_count=%d",
            len(due_sources),
        )

        for source in due_sources:
            if not self._running:
                break

            await self._sync_one(
                source, svc=svc, coordinator=coordinator, now=now
            )

    async def _sync_one(
        self,
        source,
        *,
        svc,
        coordinator,
        now,
    ) -> None:
        """Attempt to sync one source with its own session + commit.

        Each source runs in an isolated session/transaction so that
        a failure in one source cannot rollback another's committed data.

        All errors are caught and logged.  Never raises.
        """
        from app.db.session import AsyncSessionLocal

        source_id = source.id
        source_key = source.source_key
        tenant_id = source.tenant_id

        acquired = await coordinator.acquire(source_id)
        if not acquired:
            logger.info(
                "source_sync_scheduler.skip_locked source_id=%d source_key=%s",
                source_id,
                source_key,
            )
            return

        try:
            async with AsyncSessionLocal() as db:
                logger.info(
                    "source_sync_scheduler.sync_start source_id=%d "
                    "source_key=%s tenant=%s",
                    source_id,
                    source_key,
                    tenant_id,
                )

                await svc.trigger_sync(
                    db,
                    tenant_id=tenant_id,
                    source_id=source_id,
                    triggered_by="scheduler",
                )

                await db.commit()

                logger.info(
                    "source_sync_scheduler.sync_done source_id=%d source_key=%s",
                    source_id,
                    source_key,
                )

        except Exception:
            # trigger_sync already records health on failure internally,
            # so we just log here.
            logger.exception(
                "source_sync_scheduler.sync_failed source_id=%d "
                "source_key=%s",
                source_id,
                source_key,
            )

        finally:
            await coordinator.release(source_id)


# ── Singleton ────────────────────────────────────────────────────────

_scheduler: SourceSyncScheduler | None = None


def get_sync_scheduler() -> SourceSyncScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = SourceSyncScheduler()
    return _scheduler
