"""
Admin API for query history & analytics (Phase 4.2).

All endpoints are READ-ONLY and admin-gated via ``require_admin``.
system_admin sees all tenants; tenant_admin is scoped to own tenant.

🚫 No raw query text — only hashes, counts, and metadata.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import require_admin
from app.core.rbac import is_system_admin
from app.db.models.user import User
from app.db.session import get_db
from app.services.query_analytics_service import QueryAnalyticsService

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin-query-analytics"],
)

# ── singleton service ─────────────────────────────────────────────────

_svc: QueryAnalyticsService | None = None


def _get_svc() -> QueryAnalyticsService:
    global _svc
    if _svc is None:
        _svc = QueryAnalyticsService()
    return _svc


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


def _tenant_scope(admin: User, requested: str | None) -> str | None:
    """
    Enforce tenant scoping:
      * system_admin → filter by *requested* or None (all).
      * tenant_admin → always scoped to own tenant (ignore requested).
    """
    if is_system_admin(admin.role):
        return requested or None
    return admin.tenant_id


# ── response schemas ──────────────────────────────────────────────────

class _UsageItem(BaseModel):
    id: str
    created_at: str | None
    tenant_id: str
    user_id: int | None
    mode: str
    results_count: int
    tokens_query: int
    tokens_context: int
    tokens_total: int
    latency_ms: int
    query_hash: str


class _Summary(BaseModel):
    total_queries: int
    total_tokens: int
    avg_latency_ms: float


class QueryUsagePageResponse(BaseModel):
    items: list[_UsageItem]
    total: int
    summary: _Summary


class _ModeBreakdown(BaseModel):
    mode: str
    queries: int
    tokens: int
    avg_latency_ms: float


class _TenantRank(BaseModel):
    tenant_id: str
    queries: int
    tokens: int


class _UserRank(BaseModel):
    user_id: int | None
    queries: int
    tokens: int


class QueryAnalyticsResponse(BaseModel):
    summary: _Summary
    breakdown_by_mode: list[_ModeBreakdown]
    top_tenants: list[_TenantRank]
    top_users: list[_UserRank]


# ── endpoints ─────────────────────────────────────────────────────────

@router.get("/query-usages", response_model=QueryUsagePageResponse)
async def list_query_usages(
    *,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    # Query params renamed 'from_' / 'to_' to avoid Python keyword clash,
    # but the actual URL param names are 'from' and 'to'.
    from_: str | None = Query(None, alias="from", description="ISO 8601"),
    to_: str | None = Query(None, alias="to", description="ISO 8601"),
    tenant_id: str | None = Query(None),
    user_id: int | None = Query(None),
    mode: str | None = Query(None, description="vector | bm25 | hybrid"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """
    Paginated query usage history with inline summary.

    🔒 Admin-only. tenant_admin is scoped to own tenant.
    """
    svc = _get_svc()
    from_dt = _parse_iso(from_, "from")
    to_dt = _parse_iso(to_, "to")
    scoped_tenant = _tenant_scope(admin, tenant_id)

    data = await svc.get_query_usage_page(
        db,
        from_dt=from_dt,
        to_dt=to_dt,
        tenant_id=scoped_tenant,
        user_id=user_id,
        mode=mode,
        limit=limit,
        offset=offset,
    )
    return data


@router.get("/query-analytics", response_model=QueryAnalyticsResponse)
async def query_analytics(
    *,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    from_: str | None = Query(None, alias="from", description="ISO 8601"),
    to_: str | None = Query(None, alias="to", description="ISO 8601"),
    tenant_id: str | None = Query(None),
    user_id: int | None = Query(None),
    mode: str | None = Query(None, description="vector | bm25 | hybrid"),
):
    """
    Analytics overview: summary, breakdown by mode, top tenants, top users.

    🔒 Admin-only. tenant_admin is scoped to own tenant.
    """
    svc = _get_svc()
    from_dt = _parse_iso(from_, "from")
    to_dt = _parse_iso(to_, "to")
    scoped_tenant = _tenant_scope(admin, tenant_id)

    data = await svc.get_query_analytics(
        db,
        from_dt=from_dt,
        to_dt=to_dt,
        tenant_id=scoped_tenant,
        user_id=user_id,
        mode=mode,
    )
    return data
