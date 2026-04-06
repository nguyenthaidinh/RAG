"""
Monthly cost summary repository (Phase 4.3).

SQL aggregation on ``query_usages`` for per-tenant monthly cost reporting.
No raw query text is ever accessed or returned.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.query_usage import QueryUsage

logger = logging.getLogger(__name__)


class QueryCostRepository:
    """
    Read-only monthly cost aggregation on ``query_usages``.

    Every method accepts an ``AsyncSession`` from the caller.
    """

    __slots__ = ()

    async def get_monthly_cost_summary(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        year: int,
        month: int,
    ) -> dict[str, Any]:
        """
        Aggregate query usage for a specific tenant + calendar month.

        Returns::

            {
                "total_queries": int,
                "total_tokens": int,
                "avg_latency_ms": float,
            }
        """
        # Build month boundaries in UTC
        from_dt = datetime(year, month, 1, tzinfo=timezone.utc)
        if month == 12:
            to_dt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            to_dt = datetime(year, month + 1, 1, tzinfo=timezone.utc)

        stmt = (
            select(
                func.count().label("total_queries"),
                func.coalesce(func.sum(QueryUsage.tokens_total), 0).label("total_tokens"),
                func.coalesce(func.avg(QueryUsage.latency_ms), 0).label("avg_latency_ms"),
            )
            .select_from(QueryUsage)
            .where(QueryUsage.tenant_id == tenant_id)
            .where(QueryUsage.created_at >= from_dt)
            .where(QueryUsage.created_at < to_dt)
        )

        row = (await db.execute(stmt)).one()
        return {
            "total_queries": row.total_queries or 0,
            "total_tokens": int(row.total_tokens or 0),
            "avg_latency_ms": round(float(row.avg_latency_ms or 0), 1),
        }
