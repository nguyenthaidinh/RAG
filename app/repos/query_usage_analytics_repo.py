"""
Read-only analytics repository for query_usages table (Phase 4.2).

All methods use SQL aggregation — no full-dataset loading.
Stable pagination via ``ORDER BY created_at DESC, id DESC``.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Sequence

from sqlalchemy import Select, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.query_usage import QueryUsage

logger = logging.getLogger(__name__)


# ── filter helpers ────────────────────────────────────────────────────

def _apply_filters(
    stmt: Select,
    *,
    from_dt: datetime | None = None,
    to_dt: datetime | None = None,
    tenant_id: str | None = None,
    user_id: int | None = None,
    mode: str | None = None,
) -> Select:
    """Apply optional WHERE clauses shared by every analytics query."""
    if from_dt is not None:
        stmt = stmt.where(QueryUsage.created_at >= from_dt)
    if to_dt is not None:
        stmt = stmt.where(QueryUsage.created_at <= to_dt)
    if tenant_id is not None:
        stmt = stmt.where(QueryUsage.tenant_id == tenant_id)
    if user_id is not None:
        stmt = stmt.where(QueryUsage.user_id == user_id)
    if mode is not None:
        stmt = stmt.where(QueryUsage.mode == mode)
    return stmt


# ── repository ────────────────────────────────────────────────────────

class QueryUsageAnalyticsRepository:
    """
    Read-only analytics on the ``query_usages`` table.

    Every method accepts an ``AsyncSession`` from the caller (the
    service layer owns the session lifecycle).
    """

    __slots__ = ()

    async def list_query_usages(
        self,
        db: AsyncSession,
        *,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        tenant_id: str | None = None,
        user_id: int | None = None,
        mode: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[Sequence[QueryUsage], int]:
        """
        Paginated list of query usage records.

        Returns ``(items, total_count)``.

        Pagination is stable: ``ORDER BY created_at DESC, id DESC``.
        """
        # ── total count ──
        count_stmt = select(func.count()).select_from(QueryUsage)
        count_stmt = _apply_filters(
            count_stmt,
            from_dt=from_dt, to_dt=to_dt,
            tenant_id=tenant_id, user_id=user_id, mode=mode,
        )
        total = (await db.execute(count_stmt)).scalar() or 0

        # ── paginated rows ──
        rows_stmt = select(QueryUsage)
        rows_stmt = _apply_filters(
            rows_stmt,
            from_dt=from_dt, to_dt=to_dt,
            tenant_id=tenant_id, user_id=user_id, mode=mode,
        )
        rows_stmt = (
            rows_stmt
            .order_by(desc(QueryUsage.created_at), desc(QueryUsage.id))
            .limit(limit)
            .offset(offset)
        )
        result = await db.execute(rows_stmt)
        items = result.scalars().all()

        return items, total

    async def summary(
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
        Aggregate summary: total_queries, total_tokens, avg_latency_ms.
        """
        stmt = select(
            func.count().label("total_queries"),
            func.coalesce(func.sum(QueryUsage.tokens_total), 0).label("total_tokens"),
            func.coalesce(func.avg(QueryUsage.latency_ms), 0).label("avg_latency_ms"),
        ).select_from(QueryUsage)
        stmt = _apply_filters(
            stmt,
            from_dt=from_dt, to_dt=to_dt,
            tenant_id=tenant_id, user_id=user_id, mode=mode,
        )

        row = (await db.execute(stmt)).one()
        return {
            "total_queries": row.total_queries or 0,
            "total_tokens": int(row.total_tokens or 0),
            "avg_latency_ms": round(float(row.avg_latency_ms or 0), 1),
        }

    async def breakdown_by_mode(
        self,
        db: AsyncSession,
        *,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        tenant_id: str | None = None,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Per-mode breakdown: mode, queries, tokens, avg_latency_ms."""
        stmt = (
            select(
                QueryUsage.mode.label("mode"),
                func.count().label("queries"),
                func.coalesce(func.sum(QueryUsage.tokens_total), 0).label("tokens"),
                func.coalesce(func.avg(QueryUsage.latency_ms), 0).label("avg_latency_ms"),
            )
            .select_from(QueryUsage)
            .group_by(QueryUsage.mode)
            .order_by(desc(func.count()))
        )
        # Apply all filters except mode (we are grouping by mode)
        stmt = _apply_filters(
            stmt,
            from_dt=from_dt, to_dt=to_dt,
            tenant_id=tenant_id, user_id=user_id,
        )

        rows = (await db.execute(stmt)).all()
        return [
            {
                "mode": r.mode,
                "queries": r.queries,
                "tokens": int(r.tokens),
                "avg_latency_ms": round(float(r.avg_latency_ms), 1),
            }
            for r in rows
        ]

    async def top_tenants(
        self,
        db: AsyncSession,
        *,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        tenant_id: str | None = None,
        user_id: int | None = None,
        mode: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Top tenants by token consumption."""
        stmt = (
            select(
                QueryUsage.tenant_id.label("tenant_id"),
                func.count().label("queries"),
                func.coalesce(func.sum(QueryUsage.tokens_total), 0).label("tokens"),
            )
            .select_from(QueryUsage)
            .group_by(QueryUsage.tenant_id)
            .order_by(desc(func.sum(QueryUsage.tokens_total)))
            .limit(limit)
        )
        stmt = _apply_filters(
            stmt,
            from_dt=from_dt, to_dt=to_dt,
            tenant_id=tenant_id, user_id=user_id, mode=mode,
        )

        rows = (await db.execute(stmt)).all()
        return [
            {
                "tenant_id": r.tenant_id,
                "queries": r.queries,
                "tokens": int(r.tokens),
            }
            for r in rows
        ]

    async def top_users(
        self,
        db: AsyncSession,
        *,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
        tenant_id: str | None = None,
        user_id: int | None = None,
        mode: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Top users by token consumption."""
        stmt = (
            select(
                QueryUsage.user_id.label("user_id"),
                func.count().label("queries"),
                func.coalesce(func.sum(QueryUsage.tokens_total), 0).label("tokens"),
            )
            .select_from(QueryUsage)
            .where(QueryUsage.user_id.isnot(None))
            .group_by(QueryUsage.user_id)
            .order_by(desc(func.sum(QueryUsage.tokens_total)))
            .limit(limit)
        )
        stmt = _apply_filters(
            stmt,
            from_dt=from_dt, to_dt=to_dt,
            tenant_id=tenant_id, user_id=user_id, mode=mode,
        )

        rows = (await db.execute(stmt)).all()
        return [
            {
                "user_id": r.user_id,
                "queries": r.queries,
                "tokens": int(r.tokens),
            }
            for r in rows
        ]
