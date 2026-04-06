"""
Query analytics service (Phase 4.2).

Normalizes filters, enforces bounds, and combines repository calls
into structured DTOs for the admin API / dashboard.

🚫 Never exposes or stores raw query text.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.query_usage import QueryUsage
from app.repos.query_usage_analytics_repo import QueryUsageAnalyticsRepository

logger = logging.getLogger(__name__)

# ── bounds ────────────────────────────────────────────────────────────
_DEFAULT_WINDOW_DAYS = 7
_MAX_WINDOW_DAYS = 90
_MAX_LIMIT = 200
_DEFAULT_LIMIT = 50


class QueryAnalyticsService:
    """
    Read-only analytics façade on top of ``QueryUsageAnalyticsRepository``.

    Responsibilities:
      * Normalise & validate filter inputs.
      * Default time range = last 7 days.
      * Enforce max window = 90 days.
      * Cap ``limit`` to 200.
    """

    __slots__ = ("_repo",)

    def __init__(self, repo: QueryUsageAnalyticsRepository | None = None) -> None:
        self._repo = repo or QueryUsageAnalyticsRepository()

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _normalise_dates(
        from_dt: datetime | None,
        to_dt: datetime | None,
    ) -> tuple[datetime, datetime]:
        """
        Apply defaults and enforce the 90-day window cap.

        Returns ``(from_dt, to_dt)`` — both timezone-aware (UTC).
        """
        now = datetime.now(timezone.utc)

        if to_dt is None:
            to_dt = now
        elif to_dt.tzinfo is None:
            to_dt = to_dt.replace(tzinfo=timezone.utc)

        if from_dt is None:
            from_dt = to_dt - timedelta(days=_DEFAULT_WINDOW_DAYS)
        elif from_dt.tzinfo is None:
            from_dt = from_dt.replace(tzinfo=timezone.utc)

        # Enforce max window
        if (to_dt - from_dt).days > _MAX_WINDOW_DAYS:
            from_dt = to_dt - timedelta(days=_MAX_WINDOW_DAYS)

        return from_dt, to_dt

    @staticmethod
    def _cap_limit(limit: int | None) -> int:
        if limit is None or limit < 1:
            return _DEFAULT_LIMIT
        return min(limit, _MAX_LIMIT)

    @staticmethod
    def _cap_offset(offset: int | None) -> int:
        if offset is None or offset < 0:
            return 0
        return offset

    @staticmethod
    def _mode_or_none(mode: str | None) -> str | None:
        if mode and mode.strip():
            return mode.strip().lower()
        return None

    # ── public API ────────────────────────────────────────────────────

    async def get_query_usage_page(
        self,
        db: AsyncSession,
        *,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        tenant_id: str | None = None,
        user_id: int | None = None,
        mode: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        """
        Paginated query history + inline summary.

        Returns::

            {
              "items": [...],
              "total": int,
              "summary": { "total_queries", "total_tokens", "avg_latency_ms" },
            }
        """
        from_dt, to_dt = self._normalise_dates(from_dt, to_dt)
        limit_val = self._cap_limit(limit)
        offset_val = self._cap_offset(offset)
        mode_val = self._mode_or_none(mode)

        filt = dict(
            from_dt=from_dt, to_dt=to_dt,
            tenant_id=tenant_id, user_id=user_id, mode=mode_val,
        )

        items, total = await self._repo.list_query_usages(
            db, **filt, limit=limit_val, offset=offset_val,
        )
        summary = await self._repo.summary(db, **filt)

        return {
            "items": [self._serialise_usage(u) for u in items],
            "total": total,
            "summary": summary,
        }

    async def get_query_analytics(
        self,
        db: AsyncSession,
        *,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        tenant_id: str | None = None,
        user_id: int | None = None,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """
        Full analytics overview: summary + breakdown + top tenants/users.
        """
        from_dt, to_dt = self._normalise_dates(from_dt, to_dt)
        mode_val = self._mode_or_none(mode)

        filt = dict(
            from_dt=from_dt, to_dt=to_dt,
            tenant_id=tenant_id, user_id=user_id,
        )

        summary = await self._repo.summary(db, **filt, mode=mode_val)
        breakdown = await self._repo.breakdown_by_mode(db, **filt)
        top_t = await self._repo.top_tenants(db, **filt, mode=mode_val, limit=10)
        top_u = await self._repo.top_users(db, **filt, mode=mode_val, limit=10)

        return {
            "summary": summary,
            "breakdown_by_mode": breakdown,
            "top_tenants": top_t,
            "top_users": top_u,
        }

    async def get_dashboard_kpis(
        self,
        db: AsyncSession,
        *,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """
        KPIs for the dashboard home: 24h + 7d queries/tokens, avg latency,
        top tenant by tokens.
        """
        now = datetime.now(timezone.utc)

        summary_24h = await self._repo.summary(
            db, from_dt=now - timedelta(hours=24), to_dt=now,
            tenant_id=tenant_id,
        )
        summary_7d = await self._repo.summary(
            db, from_dt=now - timedelta(days=7), to_dt=now,
            tenant_id=tenant_id,
        )
        top_t = await self._repo.top_tenants(
            db, from_dt=now - timedelta(days=7), to_dt=now,
            limit=1,
        )

        return {
            "queries_24h": summary_24h["total_queries"],
            "queries_7d": summary_7d["total_queries"],
            "tokens_24h": summary_24h["total_tokens"],
            "tokens_7d": summary_7d["total_tokens"],
            "avg_latency_24h": summary_24h["avg_latency_ms"],
            "avg_latency_7d": summary_7d["avg_latency_ms"],
            "top_tenant": top_t[0] if top_t else None,
        }

    # ── serialisation (privacy-safe) ──────────────────────────────────

    @staticmethod
    def _serialise_usage(u: QueryUsage) -> dict[str, Any]:
        """
        Convert a ``QueryUsage`` ORM object to a JSON-safe dict.

        🚫 No raw query text — only hash, counts, and metadata.
        """
        return {
            "id": str(u.id),
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "tenant_id": u.tenant_id,
            "user_id": u.user_id,
            "mode": u.mode,
            "results_count": u.results_count,
            "tokens_query": u.tokens_query,
            "tokens_context": u.tokens_context,
            "tokens_total": u.tokens_total,
            "latency_ms": u.latency_ms,
            "query_hash": u.query_hash,
        }
