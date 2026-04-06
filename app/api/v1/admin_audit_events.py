"""
Admin API for audit events (Phase 6.0).

READ-ONLY — append-only audit trail.

Endpoint:
  GET /api/v1/admin/audit-events

RBAC:
  system_admin  → can view all tenants
  tenant_admin  → forced to own tenant
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import require_admin
from app.core.config import settings
from app.core.rbac import is_system_admin
from app.db.models.user import User
from app.db.session import get_db
from app.repos.audit_event_repo import AuditEventFilters, AuditEventRepository

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin-audit"],
)

_audit_repo = AuditEventRepository()


# ── Schemas ───────────────────────────────────────────────────────────

class AuditEventResponse(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    created_at: datetime
    event_type: str
    tenant_id: str
    user_id: int | None
    actor: str
    severity: str
    ref_type: str | None
    ref_id: str | None
    metadata_json: dict[str, Any]


class AuditEventsListResponse(BaseModel):
    items: list[AuditEventResponse]
    total: int


# ── Endpoint ──────────────────────────────────────────────────────────

@router.get("/audit-events", response_model=AuditEventsListResponse)
async def list_audit_events(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    # Query params — all optional
    from_dt: datetime | None = Query(None, alias="from"),
    to_dt: datetime | None = Query(None, alias="to"),
    tenant_id: str | None = Query(None),
    user_id: int | None = Query(None),
    event_type: str | None = Query(None),
    severity: str | None = Query(None),
    limit: int = Query(settings.AUDIT_PAGE_LIMIT_DEFAULT, ge=1),
    offset: int = Query(0, ge=0),
):
    """
    List audit events with filters.

    🔒 Admin-only.
      - system_admin: can filter any tenant.
      - tenant_admin: forced to own tenant (tenant_id param ignored).
    """
    # ── RBAC: scope tenant ────────────────────────────────────────
    if is_system_admin(admin.role):
        scoped_tenant_id = tenant_id  # may be None (all tenants)
    else:
        # tenant_admin: always forced to own tenant
        scoped_tenant_id = admin.tenant_id

    # ── Bounded window defaults ───────────────────────────────────
    now = datetime.now(timezone.utc)

    if from_dt is None and to_dt is None:
        # Default: last AUDIT_DEFAULT_WINDOW_DAYS
        from_dt = now - timedelta(days=settings.AUDIT_DEFAULT_WINDOW_DAYS)
        to_dt = now

    if from_dt is not None and to_dt is not None:
        # Cap max window
        max_delta = timedelta(days=settings.AUDIT_MAX_WINDOW_DAYS)
        if (to_dt - from_dt) > max_delta:
            from_dt = to_dt - max_delta

    # ── Cap limit ────────────────────────────────────────────────
    if limit > settings.AUDIT_PAGE_LIMIT_MAX:
        limit = settings.AUDIT_PAGE_LIMIT_MAX

    # ── Query ────────────────────────────────────────────────────
    filters = AuditEventFilters(
        from_dt=from_dt,
        to_dt=to_dt,
        tenant_id=scoped_tenant_id,
        user_id=user_id,
        event_type=event_type,
        severity=severity,
    )

    items, total = await _audit_repo.list_events(db, filters, limit=limit, offset=offset)

    return AuditEventsListResponse(
        items=[
            AuditEventResponse(
                id=str(e.id),
                created_at=e.created_at,
                event_type=e.event_type,
                tenant_id=e.tenant_id,
                user_id=e.user_id,
                actor=e.actor,
                severity=e.severity,
                ref_type=e.ref_type,
                ref_id=e.ref_id,
                metadata_json=e.metadata_json or {},
            )
            for e in items
        ],
        total=total,
    )
