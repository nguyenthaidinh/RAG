"""
Graceful shutdown manager (Phase 8.0).

Provides:
  - In-flight request tracking
  - Drain semantics: stop accepting, wait for in-flight, then close
  - Bounded wait (SHUTDOWN_WAIT_SECONDS)
  - DB pool cleanup

Rules:
  - The server MUST stop accepting new requests before waiting for in-flight
  - No forced task cancellation unless timeout exceeded
  - No partial writes to TokenLedger or audit tables
  - Integrates with FastAPI lifespan events
  - Gated behind GRACEFUL_SHUTDOWN_ENABLED flag
"""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from app.core.config import settings

logger = logging.getLogger(__name__)


class ShutdownManager:
    """
    Tracks in-flight requests and manages graceful drain.

    Usage:
      - Call ``request_started()`` / ``request_finished()`` around each request
      - On shutdown, call ``drain()`` to wait for in-flight to complete
    """

    __slots__ = ("_in_flight", "_draining", "_drain_event", "_start_time")

    def __init__(self) -> None:
        self._in_flight: int = 0
        self._draining: bool = False
        self._drain_event: asyncio.Event = asyncio.Event()
        self._drain_event.set()  # Initially no drain needed
        self._start_time: float = time.monotonic()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def is_draining(self) -> bool:
        return self._draining

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def request_started(self) -> bool:
        """
        Register a new in-flight request.

        Returns False if the server is draining (new requests should be rejected).
        """
        if self._draining:
            return False
        self._in_flight += 1
        self._drain_event.clear()
        return True

    def request_finished(self) -> None:
        """Mark an in-flight request as completed."""
        self._in_flight = max(0, self._in_flight - 1)
        if self._in_flight == 0:
            self._drain_event.set()

    async def drain(self, timeout: float | None = None) -> bool:
        """
        Enter drain mode and wait for all in-flight requests to finish.

        Returns True if all requests completed within timeout.
        Returns False if timeout was exceeded.
        """
        if timeout is None:
            timeout = settings.SHUTDOWN_WAIT_SECONDS

        self._draining = True
        logger.info(
            "shutdown.drain_started in_flight=%d timeout=%ds",
            self._in_flight, timeout,
        )

        if self._in_flight == 0:
            logger.info("shutdown.drain_complete immediate (0 in-flight)")
            return True

        try:
            await asyncio.wait_for(self._drain_event.wait(), timeout=timeout)
            logger.info("shutdown.drain_complete all requests finished")
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "shutdown.drain_timeout remaining_in_flight=%d",
                self._in_flight,
            )
            return False

    def reset(self) -> None:
        """Reset state (for testing only)."""
        self._in_flight = 0
        self._draining = False
        self._drain_event.set()
        self._start_time = time.monotonic()


# ── Global singleton ──────────────────────────────────────────────────

_shutdown_manager: ShutdownManager | None = None


def get_shutdown_manager() -> ShutdownManager:
    """Return the global ShutdownManager singleton."""
    global _shutdown_manager
    if _shutdown_manager is None:
        _shutdown_manager = ShutdownManager()
    return _shutdown_manager


def reset_shutdown_manager() -> None:
    """Reset the global singleton (for testing only)."""
    global _shutdown_manager
    if _shutdown_manager is not None:
        _shutdown_manager.reset()
    _shutdown_manager = None


# ── Lifespan integration ──────────────────────────────────────────────

async def shutdown_sequence() -> None:
    """
    Execute the full shutdown sequence.

    1. Drain in-flight requests
    2. Close DB engine
    """
    if not settings.GRACEFUL_SHUTDOWN_ENABLED:
        logger.info("shutdown.skipped GRACEFUL_SHUTDOWN_ENABLED=false")
        return

    mgr = get_shutdown_manager()

    # 1. Drain
    await mgr.drain()

    # 2. Close DB pool
    try:
        from app.db.session import engine
        await engine.dispose()
        logger.info("shutdown.db_pool_closed")
    except Exception:
        logger.warning("shutdown.db_pool_close_failed", exc_info=True)

    logger.info("shutdown.complete")
