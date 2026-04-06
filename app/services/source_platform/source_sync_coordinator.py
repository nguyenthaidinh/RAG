"""
Source sync coordinator (Phase 7 — Scheduled Sync).

Process-level in-memory lock guard that prevents the same source from
running two syncs simultaneously within this process.

NOT a distributed lock.  Sufficient for the single-process runtime
model where admin trigger + scheduler coexist in the same event loop.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class SourceSyncCoordinator:
    """In-memory sync lock per source_id."""

    def __init__(self) -> None:
        self._running: set[int] = set()
        self._lock = asyncio.Lock()

    async def acquire(self, source_id: int) -> bool:
        """Try to acquire sync lock for *source_id*.

        Returns ``True`` if acquired, ``False`` if already running.
        """
        async with self._lock:
            if source_id in self._running:
                return False
            self._running.add(source_id)
            return True

    async def release(self, source_id: int) -> None:
        """Release sync lock for *source_id*."""
        async with self._lock:
            self._running.discard(source_id)

    @asynccontextmanager
    async def sync_lock(self, source_id: int) -> AsyncIterator[bool]:
        """Context manager for acquire/release.

        Yields ``True`` if lock was acquired, ``False`` otherwise.
        On ``False`` the caller should skip the sync.
        """
        acquired = await self.acquire(source_id)
        try:
            yield acquired
        finally:
            if acquired:
                await self.release(source_id)

    def is_running(self, source_id: int) -> bool:
        """Check if a source is currently syncing (non-blocking)."""
        return source_id in self._running


# ── Singleton ────────────────────────────────────────────────────────

_coordinator: SourceSyncCoordinator | None = None


def get_sync_coordinator() -> SourceSyncCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = SourceSyncCoordinator()
    return _coordinator
