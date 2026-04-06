"""
Tenant management API – system_admin only.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.core.auth_deps import require_system_admin
from app.db.models.user import User
from app.db.models.tenant import Tenant
from app.db.models.tenant_quota import TenantQuota
from app.services.quota_service import QuotaService
from app.core.audit import audit_log

router = APIRouter(prefix="/api/v1/admin/tenants", tags=["admin-tenants"])


class CreateTenantRequest(BaseModel):
    id: str
    name: str
    max_users: int = 10
    max_requests: int = 10_000
    max_tokens: int = 10_000_000
    max_storage_mb: int = 1024


class TenantResponse(BaseModel):
    id: str
    name: str
    is_active: bool
    max_users: int
    created_at: str

    class Config:
        from_attributes = True


class UpdateTenantRequest(BaseModel):
    name: str | None = None
    is_active: bool | None = None
    max_users: int | None = None


@router.get("/", response_model=list[TenantResponse])
async def list_tenants(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_system_admin),
):
    """List all tenants (system_admin only)."""
    res = await db.execute(select(Tenant).order_by(Tenant.created_at.desc()))
    tenants = res.scalars().all()
    return [
        TenantResponse(
            id=t.id,
            name=t.name,
            is_active=t.is_active,
            max_users=t.max_users,
            created_at=t.created_at.isoformat(),
        )
        for t in tenants
    ]


@router.post("/", response_model=TenantResponse, status_code=201)
async def create_tenant(
    data: CreateTenantRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_system_admin),
):
    """Create a new tenant with a default quota (system_admin only)."""
    # Check if tenant already exists
    existing = await db.execute(select(Tenant).where(Tenant.id == data.id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Tenant '{data.id}' already exists")

    tenant = Tenant(
        id=data.id,
        name=data.name,
        is_active=True,
        max_users=data.max_users,
    )
    db.add(tenant)
    await db.flush()

    # Create tenant quota
    await QuotaService.create_tenant_quota(
        db,
        tenant_id=tenant.id,
        max_requests=data.max_requests,
        max_tokens=data.max_tokens,
        max_storage_mb=data.max_storage_mb,
    )

    await db.commit()

    audit_log(
        action="tenant.create",
        actor_user_id=admin.id,
        tenant_id=tenant.id,
        target_id=tenant.id,
        detail=f"Created tenant '{data.name}' (max_users={data.max_users})",
    )

    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        is_active=tenant.is_active,
        max_users=tenant.max_users,
        created_at=tenant.created_at.isoformat(),
    )


@router.patch("/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: str,
    data: UpdateTenantRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_system_admin),
):
    """Update tenant properties (system_admin only)."""
    tenant = (
        await db.execute(select(Tenant).where(Tenant.id == tenant_id))
    ).scalar_one_or_none()
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    if data.name is not None:
        tenant.name = data.name
    if data.is_active is not None:
        tenant.is_active = data.is_active
    if data.max_users is not None:
        tenant.max_users = data.max_users

    await db.commit()

    audit_log(
        action="tenant.update",
        actor_user_id=admin.id,
        tenant_id=tenant.id,
        target_id=tenant.id,
        detail=f"Updated tenant: name={tenant.name}, is_active={tenant.is_active}, max_users={tenant.max_users}",
    )

    return TenantResponse(
        id=tenant.id,
        name=tenant.name,
        is_active=tenant.is_active,
        max_users=tenant.max_users,
        created_at=tenant.created_at.isoformat(),
    )
