from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# DTO
# ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class UsageLogEntry:
    """
    Lightweight immutable DTO enqueued by middleware.
    Must NOT contain ORM objects or sessions.
    """
    request_id: str
    user_id: int
    tenant_id: str
    endpoint: str
    method: str
    status_code: int
    success: bool
    api_key_id: int | None = None
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0
    file_size_bytes: int = 0
    request_cost: float = 0.0


# ─────────────────────────────────────────────────────────────────────
# Dispatcher (Fixed Worker Pool – Production Safe)
# ─────────────────────────────────────────────────────────────────────

class UsageLogDispatcher:
    """
    Bounded queue + fixed worker pool.

    Guarantees:
      - No unbounded task creation
      - No memory explosion under burst traffic
      - Strict DB timeout per write
      - Fail-open (never breaks request path)
      - Graceful shutdown with bounded drain
    """

    __slots__ = (
        "_queue",
        "_workers",
        "_running",
        "_dropped",
        "_processed",
        "_timeouts",
        "_db_timeout",
        "_drain_timeout",
        "_concurrency",
    )

    def __init__(
        self,
        maxsize: int | None = None,
        concurrency: int | None = None,
        db_timeout: float | None = None,
        drain_timeout: float | None = None,
    ) -> None:
        maxsize = settings.USAGE_LOG_QUEUE_MAXSIZE if maxsize is None else maxsize
        concurrency = settings.USAGE_LOG_CONCURRENCY if concurrency is None else concurrency
        db_timeout = settings.USAGE_LOG_DB_TIMEOUT_SEC if db_timeout is None else db_timeout
        drain_timeout = settings.USAGE_LOG_DRAIN_TIMEOUT_SEC if drain_timeout is None else drain_timeout

        # Ensure sane bounds
        if concurrency <= 0:
            concurrency = 1
        if maxsize <= 0:
            maxsize = 1

        self._queue: asyncio.Queue[UsageLogEntry] = asyncio.Queue(maxsize=maxsize)
        self._workers: list[asyncio.Task] = []
        self._running = False

        self._dropped = 0
        self._processed = 0
        self._timeouts = 0

        self._db_timeout = float(db_timeout)
        self._drain_timeout = float(drain_timeout)
        self._concurrency = int(concurrency)

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def enqueue(self, entry: UsageLogEntry) -> bool:
        """
        Non-blocking enqueue.
        Returns False if queue is full (fail-open behavior).
        """
        try:
            self._queue.put_nowait(entry)
            return True
        except asyncio.QueueFull:
            self._dropped += 1
            logger.warning(
                "usage.drop_queue_full dropped=%d qsize=%d max=%d",
                self._dropped,
                self._queue.qsize(),
                self._queue.maxsize,
            )
            return False

    async def start(self) -> None:
        """
        Start fixed worker pool.
        Idempotent: safe to call multiple times.
        """
        if self._running:
            return

        self._running = True

        # Defensive: ensure no stale workers
        if self._workers:
            for t in self._workers:
                t.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

        for i in range(self._concurrency):
            self._workers.append(asyncio.create_task(self._worker(i)))

        logger.info(
            "usage_dispatcher.started maxsize=%d workers=%d db_timeout=%.1fs drain_timeout=%.1fs",
            self._queue.maxsize,
            self._concurrency,
            self._db_timeout,
            self._drain_timeout,
        )

    async def stop(self) -> None:
        """
        Stop dispatcher:
          1) Stop accepting new work
          2) Wait bounded time for queue to drain
          3) Cancel workers
        Idempotent: safe to call multiple times.
        """
        if not self._running and not self._workers:
            return

        self._running = False

        # Attempt bounded drain
        try:
            await asyncio.wait_for(self._queue.join(), timeout=self._drain_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "usage_dispatcher.drain_timeout remaining=%d",
                self._queue.qsize(),
            )

        # Cancel workers (best-effort)
        for task in self._workers:
            task.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info(
            "usage_dispatcher.stopped processed=%d dropped=%d timeouts=%d remaining=%d",
            self._processed,
            self._dropped,
            self._timeouts,
            self._queue.qsize(),
        )

    # ─────────────────────────────────────────────────────────
    # Worker Loop
    # ─────────────────────────────────────────────────────────

    async def _worker(self, worker_id: int) -> None:
        """
        Fixed worker consuming queue.
        Uses timeout to re-check _running state (no dead wait).
        """
        while self._running or not self._queue.empty():
            try:
                entry = await asyncio.wait_for(self._queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._write_one(entry)
            finally:
                # Must always task_done() to unblock queue.join()
                self._queue.task_done()

    # ─────────────────────────────────────────────────────────
    # DB Write Logic
    # ─────────────────────────────────────────────────────────

    async def _write_one(self, entry: UsageLogEntry) -> None:
        try:
            await asyncio.wait_for(self._do_db_insert(entry), timeout=self._db_timeout)
            self._processed += 1

        except asyncio.TimeoutError:
            self._timeouts += 1
            logger.warning(
                "usage.db_timeout tenant_id=%s request_id=%s timeout=%.1fs",
                entry.tenant_id,
                entry.request_id,
                self._db_timeout,
            )

        except Exception:
            logger.error(
                "usage.write_failed tenant_id=%s request_id=%s",
                entry.tenant_id,
                entry.request_id,
                exc_info=True,
            )

    async def _do_db_insert(self, entry: UsageLogEntry) -> None:
        """
        Isolated DB write. New session per entry.
        """
        from app.db.session import AsyncSessionLocal
        from app.services.usage_service import UsageService

        async with AsyncSessionLocal() as db:
            await UsageService.log_usage(
                db=db,
                request_id=entry.request_id,
                user_id=entry.user_id,
                tenant_id=entry.tenant_id,
                endpoint=entry.endpoint,
                method=entry.method,
                status_code=entry.status_code,
                success=entry.success,
                api_key_id=entry.api_key_id,
                tokens_input=entry.tokens_input,
                tokens_output=entry.tokens_output,
                tokens_total=entry.tokens_total,
                file_size_bytes=entry.file_size_bytes,
                request_cost=entry.request_cost,
            )

    # ─────────────────────────────────────────────────────────
    # Observability / Stats
    # ─────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "queue_size": self._queue.qsize(),
            "queue_max": self._queue.maxsize,
            "processed": self._processed,
            "dropped": self._dropped,
            "timeouts": self._timeouts,
            "running": self._running,
            "workers": self._concurrency,
            "remaining": self._queue.qsize(),
        }


# ─────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────

_dispatcher: UsageLogDispatcher | None = None


def get_usage_dispatcher() -> UsageLogDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = UsageLogDispatcher()
    return _dispatcher


def reset_usage_dispatcher() -> None:
    """
    For testing only.
    """
    global _dispatcher
    _dispatcher = None
