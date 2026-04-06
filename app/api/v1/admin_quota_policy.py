"""
Admin API for managing plans and tenant quota settings (Phase 5.0).

Endpoints:
  GET  /api/v1/admin/plans              — list all plans
  PUT  /api/v1/admin/plans/{code}       — upsert a plan
  GET  /api/v1/admin/tenants/{tenant_id}/settings  — get tenant settings
  PUT  /api/v1/admin/tenants/{tenant_id}/settings  — upsert tenant settings
"""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import require_admin
from app.core.rbac import is_system_admin
from app.db.models.user import User
from app.db.session import get_db
from app.repos.plan_repo import PlanRepository
from app.repos.tenant_setting_repo import TenantSettingRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/admin",
    tags=["admin-quota-policy"],
)

_plan_repo = PlanRepository()
_setting_repo = TenantSettingRepository()


# ── Schemas ───────────────────────────────────────────────────────────

class PlanResponse(BaseModel):
    model_config = {"from_attributes": True}

    code: str
    name: str
    is_active: bool
    limits_json: dict[str, Any]


class UpsertPlanRequest(BaseModel):
    name: str
    limits_json: dict[str, Any]
    is_active: bool = True


class TenantSettingResponse(BaseModel):
    model_config = {"from_attributes": True}

    tenant_id: str
    plan_code: str
    quota_overrides_json: dict[str, Any] | None
    enforce_user_rate_limit: bool


class UpsertTenantSettingRequest(BaseModel):
    plan_code: str | None = None
    quota_overrides_json: dict[str, Any] | None = None
    enforce_user_rate_limit: bool | None = None


# ── Helpers ───────────────────────────────────────────────────────────

def _enforce_tenant_scope(admin: User, tenant_id: str) -> None:
    """
    Enforce tenant scoping: system_admin can access any tenant,
    tenant_admin can only access their own.
    """
    if is_system_admin(admin.role):
        return
    if admin.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Access denied to this tenant")


# ── Plan Endpoints ────────────────────────────────────────────────────

@router.get("/plans", response_model=list[PlanResponse])
async def list_plans(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """List all plans. Admin-only."""
    plans = await _plan_repo.list_all(db)
    return [
        PlanResponse(
            code=p.code,
            name=p.name,
            is_active=p.is_active,
            limits_json=p.limits_json,
        )
        for p in plans
    ]


@router.put("/plans/{code}", response_model=PlanResponse)
async def upsert_plan(
    code: str,
    data: UpsertPlanRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Create or update a plan. Admin-only (system_admin recommended)."""
    if not is_system_admin(admin.role):
        raise HTTPException(status_code=403, detail="Only system admins can manage plans")

    plan = await _plan_repo.upsert_plan(
        db,
        code=code,
        name=data.name,
        limits_json=data.limits_json,
        is_active=data.is_active,
    )
    await db.commit()

    # ── Phase 6.0: emit PLAN_CHANGED audit event ──────────────
    try:
        from app.services.audit_service import get_audit_service
        request_id = getattr(request.state, "request_id", None)
        await get_audit_service().log_plan_changed(
            db,
            tenant_id=admin.tenant_id,
            user_id=admin.id,
            request_id=request_id,
            plan_code=code,
        )
    except Exception:
        logger.warning("audit.plan_changed_failed", exc_info=True)

    return PlanResponse(
        code=plan.code,
        name=plan.name,
        is_active=plan.is_active,
        limits_json=plan.limits_json,
    )


# ── Tenant Settings Endpoints ────────────────────────────────────────

@router.get("/tenants/{tenant_id}/settings", response_model=TenantSettingResponse)
async def get_tenant_settings(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Get tenant quota/plan settings. tenant_admin scoped to own tenant."""
    _enforce_tenant_scope(admin, tenant_id)

    setting = await _setting_repo.get(db, tenant_id)
    if setting is None:
        # Return defaults
        return TenantSettingResponse(
            tenant_id=tenant_id,
            plan_code="free",
            quota_overrides_json=None,
            enforce_user_rate_limit=False,
        )

    return TenantSettingResponse(
        tenant_id=setting.tenant_id,
        plan_code=setting.plan_code,
        quota_overrides_json=setting.quota_overrides_json,
        enforce_user_rate_limit=setting.enforce_user_rate_limit,
    )


@router.put("/tenants/{tenant_id}/settings", response_model=TenantSettingResponse)
async def upsert_tenant_settings(
    tenant_id: str,
    data: UpsertTenantSettingRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Update tenant quota/plan settings. tenant_admin scoped to own tenant."""
    _enforce_tenant_scope(admin, tenant_id)

    # If plan_code is provided, validate it exists
    if data.plan_code is not None:
        plan = await _plan_repo.get_by_code(db, data.plan_code)
        if plan is None:
            raise HTTPException(status_code=404, detail=f"Plan '{data.plan_code}' not found")

    setting = await _setting_repo.upsert(
        db,
        tenant_id=tenant_id,
        plan_code=data.plan_code,
        quota_overrides_json=data.quota_overrides_json,
        enforce_user_rate_limit=data.enforce_user_rate_limit,
    )
    await db.commit()

    # ── Phase 6.0: emit TENANT_QUOTA_OVERRIDE audit event ─────
    try:
        from app.services.audit_service import get_audit_service
        request_id = getattr(request.state, "request_id", None)
        override_keys = list(data.quota_overrides_json.keys()) if data.quota_overrides_json else []
        await get_audit_service().log_tenant_quota_override(
            db,
            tenant_id=admin.tenant_id,
            user_id=admin.id,
            request_id=request_id,
            plan_code=data.plan_code,
            override_changed=data.quota_overrides_json is not None,
            override_keys_changed=override_keys,
            target_tenant_id=tenant_id,
        )
    except Exception:
        logger.warning("audit.tenant_quota_override_failed", exc_info=True)

    return TenantSettingResponse(
        tenant_id=setting.tenant_id,
        plan_code=setting.plan_code,
        quota_overrides_json=setting.quota_overrides_json,
        enforce_user_rate_limit=setting.enforce_user_rate_limit,
    )
