"""
Backpressure & load shedding (Phase 8.0).

Provides:
  - Global concurrency limiter for expensive operations
  - Per-tenant concurrency limiter (tenant isolation)
  - Bounded queue depth tracking
  - Fail-fast (429/503) on overload — no infinite queuing

Rules:
  - Prefer fail-fast over resource exhaustion
  - Do NOT queue infinitely
  - Dropped work is silent & safe (best-effort)
  - One abusive tenant MUST NOT degrade the whole system
  - Gated behind BACKPRESSURE_ENABLED flag
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)


class BackpressureManager:
    """
    Manage global + per-tenant concurrency limits.

    - Global semaphore bounds total concurrent expensive ops.
    - Per-tenant counters prevent a single tenant from monopolizing resources.
    - ``acquire()`` context manager raises 429/503 on overload.
    """

    __slots__ = (
        "_global_semaphore",
        "_global_max",
        "_per_tenant_max",
        "_tenant_counts",
        "_total_rejected",
        "_total_admitted",
    )

    def __init__(
        self,
        global_max: int | None = None,
        per_tenant_max: int | None = None,
    ) -> None:
        self._global_max = global_max or settings.BACKPRESSURE_MAX_CONCURRENT_GLOBAL
        self._per_tenant_max = per_tenant_max or settings.BACKPRESSURE_MAX_CONCURRENT_PER_TENANT
        self._global_semaphore = asyncio.Semaphore(self._global_max)
        self._tenant_counts: dict[str, int] = defaultdict(int)
        self._total_rejected: int = 0
        self._total_admitted: int = 0

    @property
    def global_max(self) -> int:
        return self._global_max

    @property
    def per_tenant_max(self) -> int:
        return self._per_tenant_max

    @property
    def current_global(self) -> int:
        """Current number of globally held slots."""
        return self._global_max - self._global_semaphore._value

    @property
    def total_rejected(self) -> int:
        return self._total_rejected

    @property
    def total_admitted(self) -> int:
        return self._total_admitted

    def tenant_current(self, tenant_id: str) -> int:
        """Current concurrent ops for a specific tenant."""
        return self._tenant_counts.get(tenant_id, 0)

    @asynccontextmanager
    async def acquire(self, tenant_id: str) -> AsyncGenerator[None, None]:
        """
        Acquire a slot for an expensive operation.

        Raises HTTPException(429) if per-tenant limit exceeded.
        Raises HTTPException(503) if global limit exceeded.
        """
        if not settings.BACKPRESSURE_ENABLED:
            yield
            return

        # ── Per-tenant check (fail-fast) ──
        if self._tenant_counts[tenant_id] >= self._per_tenant_max:
            self._total_rejected += 1
            logger.warning(
                "backpressure.tenant_rejected tenant_id=%s current=%d limit=%d",
                tenant_id, self._tenant_counts[tenant_id], self._per_tenant_max,
            )
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "tenant_concurrency_exceeded",
                    "message": "Too many concurrent requests for this tenant. Please retry.",
                },
            )

        # ── Global check (fail-fast, non-blocking) ──
        acquired = self._global_semaphore._value > 0
        if not acquired:
            self._total_rejected += 1
            logger.warning(
                "backpressure.global_rejected current=%d limit=%d",
                self.current_global, self._global_max,
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "server_overloaded",
                    "message": "Server is at capacity. Please retry shortly.",
                },
            )

        # Acquire the global semaphore
        await self._global_semaphore.acquire()
        self._tenant_counts[tenant_id] += 1
        self._total_admitted += 1

        try:
            yield
        finally:
            self._tenant_counts[tenant_id] = max(0, self._tenant_counts[tenant_id] - 1)
            if self._tenant_counts[tenant_id] == 0:
                del self._tenant_counts[tenant_id]
            self._global_semaphore.release()

    def stats(self) -> dict:
        """Return current backpressure statistics (for ops dashboard)."""
        return {
            "global_current": self.current_global,
            "global_max": self._global_max,
            "per_tenant_max": self._per_tenant_max,
            "total_admitted": self._total_admitted,
            "total_rejected": self._total_rejected,
            "active_tenants": len(self._tenant_counts),
            "tenant_counts": dict(self._tenant_counts),
        }

    def reset(self) -> None:
        """Reset state (for testing only)."""
        self._global_semaphore = asyncio.Semaphore(self._global_max)
        self._tenant_counts.clear()
        self._total_rejected = 0
        self._total_admitted = 0


# ── Global singleton ──────────────────────────────────────────────────

_backpressure_manager: BackpressureManager | None = None


def get_backpressure_manager() -> BackpressureManager:
    """Return the global BackpressureManager singleton."""
    global _backpressure_manager
    if _backpressure_manager is None:
        _backpressure_manager = BackpressureManager()
    return _backpressure_manager


def reset_backpressure_manager() -> None:
    """Reset the global singleton (for testing only)."""
    global _backpressure_manager
    if _backpressure_manager is not None:
        _backpressure_manager.reset()
    _backpressure_manager = None
