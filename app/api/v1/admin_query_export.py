"""
Admin API for CSV export, monthly cost summary, and retention (Phase 4.3).

All endpoints are admin-gated via ``require_admin``.
system_admin sees all tenants; tenant_admin is scoped to own tenant.

🚫 No raw query text — only hashes, counts, and metadata.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import asc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import require_admin
from app.core.rbac import is_system_admin
from app.db.models.query_usage import QueryUsage
from app.db.models.user import User
from app.db.session import get_db
from app.services.query_cost_service import QueryCostService
from app.services.query_retention_service import QueryRetentionService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin-query-export"],
)

# ── singleton services ────────────────────────────────────────────────

_cost_svc: QueryCostService | None = None
_retention_svc: QueryRetentionService | None = None


def _get_cost_svc() -> QueryCostService:
    global _cost_svc
    if _cost_svc is None:
        _cost_svc = QueryCostService()
    return _cost_svc


def _get_retention_svc() -> QueryRetentionService:
    global _retention_svc
    if _retention_svc is None:
        _retention_svc = QueryRetentionService()
    return _retention_svc


# ── helpers ───────────────────────────────────────────────────────────

def _parse_iso(value: str | None, name: str) -> datetime | None:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Safety: never pass naive datetimes into DB queries
        if dt.tzinfo is None:
            from datetime import timezone
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {name} format — expected ISO 8601",
        )


def _parse_month(value: str | None) -> tuple[int, int] | None:
    """Parse ``YYYY-MM`` into ``(year, month)`` or return None."""
    if value is None:
        return None
    try:
        parts = value.strip().split("-")
        if len(parts) != 2:
            raise ValueError
        year, month = int(parts[0]), int(parts[1])
        if not (1 <= month <= 12):
            raise ValueError
        return year, month
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail="Invalid month format — expected YYYY-MM",
        )


def _tenant_scope(admin: User, requested: str | None) -> str | None:
    """
    Enforce tenant scoping:
      * system_admin → filter by *requested* or None (all).
      * tenant_admin → always scoped to own tenant.
    """
    if is_system_admin(admin.role):
        return requested or None
    return admin.tenant_id


def _require_tenant(admin: User, requested: str | None) -> str:
    """
    Like ``_tenant_scope`` but REQUIRES a tenant_id.
    system_admin must provide one explicitly; tenant_admin uses own.
    """
    if is_system_admin(admin.role):
        if not requested:
            raise HTTPException(
                status_code=422,
                detail="tenant_id is required for system_admin",
            )
        return requested
    return admin.tenant_id


# ── CSV column spec (privacy-safe) ────────────────────────────────────

CSV_COLUMNS = [
    "created_at",
    "tenant_id",
    "user_id",
    "mode",
    "results_count",
    "tokens_query",
    "tokens_context",
    "tokens_total",
    "latency_ms",
    "query_hash",
]

_EXPORT_BATCH_SIZE = 1000


# ── response schemas ──────────────────────────────────────────────────

class MonthlyCostResponse(BaseModel):
    tenant_id: str
    month: str
    total_queries: int
    total_tokens: int
    avg_latency_ms: float


class RetentionPurgeResponse(BaseModel):
    deleted: int
    retention_days: int
    cutoff: str


# ═══════════════════════════════════════════════════════════════════════
# 1. CSV Export
# ═══════════════════════════════════════════════════════════════════════

@router.get("/query-usages/export")
async def export_query_usages_csv(
    *,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
    from_: str | None = Query(None, alias="from", description="ISO 8601"),
    to_: str | None = Query(None, alias="to", description="ISO 8601"),
    month: str | None = Query(None, description="YYYY-MM (mutually exclusive with from/to)"),
    mode: str | None = Query(None, description="vector | bm25 | hybrid"),
):
    """
    Stream query_usages as CSV.

    🔒 Admin-only.  tenant_admin is scoped to own tenant.

    ``month`` is mutually exclusive with ``from``/``to``.
    Order: ``created_at ASC, id ASC``.
    """
    # ── mutual exclusivity ──
    if month and (from_ or to_):
        raise HTTPException(
            status_code=422,
            detail="month is mutually exclusive with from/to",
        )

    # ── date range ──
    from_dt: datetime | None = None
    to_dt: datetime | None = None

    if month:
        parsed = _parse_month(month)
        assert parsed is not None
        year, mon = parsed
        from_dt = datetime(year, mon, 1, tzinfo=timezone.utc)
        if mon == 12:
            to_dt = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            to_dt = datetime(year, mon + 1, 1, tzinfo=timezone.utc)
    else:
        from_dt = _parse_iso(from_, "from")
        to_dt = _parse_iso(to_, "to")

    # ── tenant scope ──
    scoped_tenant = _tenant_scope(admin, tenant_id)

    # ── mode normalisation ──
    mode_val = mode.strip().lower() if mode and mode.strip() else None

    # ── streaming generator ──
    async def _generate():
        # Header row
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(CSV_COLUMNS)
        yield buf.getvalue()

        offset = 0
        while True:
            stmt = select(QueryUsage)
            if from_dt is not None:
                stmt = stmt.where(QueryUsage.created_at >= from_dt)
            if to_dt is not None:
                stmt = stmt.where(QueryUsage.created_at < to_dt)
            if scoped_tenant is not None:
                stmt = stmt.where(QueryUsage.tenant_id == scoped_tenant)
            if mode_val is not None:
                stmt = stmt.where(QueryUsage.mode == mode_val)

            stmt = (
                stmt
                .order_by(asc(QueryUsage.created_at), asc(QueryUsage.id))
                .limit(_EXPORT_BATCH_SIZE)
                .offset(offset)
            )

            result = await db.execute(stmt)
            rows = result.scalars().all()

            if not rows:
                break

            for row in rows:
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow([
                    row.created_at.isoformat() if row.created_at else "",
                    row.tenant_id,
                    row.user_id if row.user_id is not None else "",
                    row.mode,
                    row.results_count,
                    row.tokens_query,
                    row.tokens_context,
                    row.tokens_total,
                    row.latency_ms,
                    row.query_hash,
                ])
                yield buf.getvalue()

            offset += len(rows)

            if len(rows) < _EXPORT_BATCH_SIZE:
                break

    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=query_usages_export.csv",
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. Monthly Cost Summary
# ═══════════════════════════════════════════════════════════════════════

@router.get("/query-costs/monthly", response_model=MonthlyCostResponse)
async def monthly_cost_summary(
    *,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    tenant_id: str | None = Query(None),
    month: str = Query(..., description="YYYY-MM"),
):
    """
    Monthly cost summary for a tenant.

    🔒 Admin-only.  tenant_admin is scoped to own tenant.
    system_admin must provide ``tenant_id``.
    """
    scoped_tenant = _require_tenant(admin, tenant_id)

    parsed = _parse_month(month)
    assert parsed is not None
    year, mon = parsed

    svc = _get_cost_svc()
    try:
        data = await svc.get_monthly_cost(
            db,
            tenant_id=scoped_tenant,
            year=year,
            month=mon,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return data


# ═══════════════════════════════════════════════════════════════════════
# 3. Retention Purge
# ═══════════════════════════════════════════════════════════════════════

@router.post("/query-usages/retention/purge", response_model=RetentionPurgeResponse)
async def purge_expired(
    *,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Purge query_usages older than the retention window.

    🔒 Admin-only (system_admin recommended).
    Operates in batches.  Safe to call repeatedly.
    """
    from app.core.config import settings as _settings

    svc = _get_retention_svc()
    now = datetime.now(timezone.utc)
    deleted = await svc.purge_expired_query_usages(db, now=now)

    cutoff = now - timedelta(days=_settings.QUERY_USAGE_RETENTION_DAYS)

    return {
        "deleted": deleted,
        "retention_days": _settings.QUERY_USAGE_RETENTION_DAYS,
        "cutoff": cutoff.isoformat(),
    }
