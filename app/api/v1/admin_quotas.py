from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.auth_deps import require_admin
from app.db.models.user import User
from app.services.quota_service import QuotaService
from app.services.user_service import UserService
from app.core.rbac import is_system_admin

router = APIRouter(prefix="/api/v1/admin/quotas", tags=["admin-quotas"])


class QuotaUpdateRequest(BaseModel):
    plan: str | None = None
    max_tokens: int | None = None
    max_requests: int | None = None
    max_storage_mb: int | None = None
    is_active: bool | None = None


class QuotaResponse(BaseModel):
    user_id: int
    plan: str
    is_active: bool
    max_tokens: int
    used_tokens: int
    max_requests: int
    used_requests: int
    max_storage_mb: int
    used_storage_mb: int
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


def _enforce_tenant_scope(admin: User, target_user: User) -> None:
    """Raise 403 if tenant_admin tries to access user outside their tenant."""
    if is_system_admin(admin.role):
        return
    if target_user.tenant_id != admin.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied: user is in another tenant")


@router.get("/{user_id}", response_model=QuotaResponse)
async def get_quota(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = await UserService.get_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    _enforce_tenant_scope(admin, user)

    quota = await QuotaService.get_by_user_id(db, user_id)
    if not quota:
        raise HTTPException(status_code=404, detail="Quota not found")

    return quota


@router.put("/{user_id}", response_model=QuotaResponse)
async def update_quota(
    user_id: int,
    data: QuotaUpdateRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = await UserService.get_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    _enforce_tenant_scope(admin, user)

    quota = await QuotaService.update_quota(
        db,
        user_id,
        plan=data.plan,
        max_tokens=data.max_tokens,
        max_requests=data.max_requests,
        max_storage_mb=data.max_storage_mb,
        is_active=data.is_active,
    )
    if not quota:
        raise HTTPException(status_code=404, detail="Quota not found")

    await db.commit()
    return quota


@router.put("/{user_id}/unlimited", response_model=QuotaResponse)
async def set_unlimited(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = await UserService.get_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    _enforce_tenant_scope(admin, user)

    quota = await QuotaService.set_unlimited(db, user_id)
    if not quota:
        raise HTTPException(status_code=404, detail="Quota not found")

    await db.commit()
    return quota


@router.get("/{user_id}/usage")
async def get_usage(
    user_id: int,
    days: int = 7,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = await UserService.get_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    _enforce_tenant_scope(admin, user)

    if days not in [1, 7, 30]:
        days = 7

    aggregates = await QuotaService.get_usage_aggregates(db, user_id, days=days)
    return aggregates
