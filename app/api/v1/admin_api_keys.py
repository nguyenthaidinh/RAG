from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.auth_deps import require_admin
from app.db.models.user import User
from app.services.user_service import UserService
from app.services.api_key_service import APIKeyService
from app.services.admin_api_key_service import AdminAPIKeyService
from app.core.rbac import is_system_admin

router = APIRouter(prefix="/api/v1/admin/api-keys", tags=["admin-api-keys"])


class CreateAPIKeyRequest(BaseModel):
    user_id: int
    name: str


class RenameAPIKeyRequest(BaseModel):
    name: str


class APIKeyResponse(BaseModel):
    id: int
    user_id: int
    tenant_id: str
    user_email: str
    user_role: str
    name: str
    prefix: str
    is_active: bool
    revoked_at: str | None
    last_used_at: str | None
    created_at: str


class CreateAPIKeyResponse(BaseModel):
    api_key: APIKeyResponse
    key: str  # Full key, only returned once


def _build_api_key_response(api_key, user) -> APIKeyResponse:
    return APIKeyResponse(
        id=api_key.id,
        user_id=api_key.user_id,
        tenant_id=api_key.tenant_id,
        user_email=user.email if user else "",
        user_role=user.role if user else "",
        name=api_key.name,
        prefix=api_key.prefix,
        is_active=api_key.is_active,
        revoked_at=api_key.revoked_at.isoformat() if api_key.revoked_at else None,
        last_used_at=api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        created_at=api_key.created_at.isoformat(),
    )


@router.get("/", response_model=list[APIKeyResponse])
async def list_api_keys(
    user_id: int | None = Query(None, description="Filter by user_id"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    List API keys. system_admin sees all; tenant_admin sees own tenant only.
    """
    tenant_filter = None
    if not is_system_admin(admin.role):
        tenant_filter = admin.tenant_id

    keys = await AdminAPIKeyService.list_api_keys(
        db, user_id=user_id, tenant_id=tenant_filter
    )
    return keys


@router.post("/", response_model=CreateAPIKeyResponse)
async def create_api_key(
    data: CreateAPIKeyRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Create API key. tenant_admin can only create for users in their own tenant.
    """
    user = await UserService.get_by_id(db, data.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Tenant scoping: tenant_admin can only create for own tenant
    if not is_system_admin(admin.role):
        if user.tenant_id != admin.tenant_id:
            raise HTTPException(
                status_code=403,
                detail="Cannot create API key for user in another tenant",
            )

    api_key, full_key = await AdminAPIKeyService.create_api_key(
        db,
        user_id=data.user_id,
        tenant_id=user.tenant_id,
        name=data.name,
        actor_user_id=admin.id,
    )
    await db.commit()

    return CreateAPIKeyResponse(
        api_key=_build_api_key_response(api_key, user),
        key=full_key,
    )


@router.post("/{key_id}/rotate", response_model=CreateAPIKeyResponse)
async def rotate_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Rotate API key. tenant_admin scoped to own tenant.
    """
    old_key = await APIKeyService.get_by_id(db, key_id)
    if not old_key:
        raise HTTPException(status_code=404, detail="API key not found")

    # Tenant scoping
    if not is_system_admin(admin.role):
        if old_key.tenant_id != admin.tenant_id:
            raise HTTPException(status_code=403, detail="Cannot rotate key in another tenant")

    result = await APIKeyService.rotate_api_key(
        db, key_id, actor_user_id=admin.id
    )
    if not result:
        raise HTTPException(status_code=404, detail="API key not found")

    api_key, full_key = result
    await db.commit()

    user = await UserService.get_by_id(db, api_key.user_id)

    return CreateAPIKeyResponse(
        api_key=_build_api_key_response(api_key, user),
        key=full_key,
    )


@router.post("/{key_id}/revoke")
async def revoke_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Revoke API key. tenant_admin scoped to own tenant.
    """
    existing_key = await APIKeyService.get_by_id(db, key_id)
    if not existing_key:
        raise HTTPException(status_code=404, detail="API key not found")

    # Tenant scoping
    if not is_system_admin(admin.role):
        if existing_key.tenant_id != admin.tenant_id:
            raise HTTPException(status_code=403, detail="Cannot revoke key in another tenant")

    api_key = await APIKeyService.revoke_api_key(
        db, key_id, actor_user_id=admin.id
    )
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    await db.commit()
    return {"ok": True}


@router.post("/{key_id}/rename")
async def rename_api_key(
    key_id: int,
    data: RenameAPIKeyRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Rename API key. tenant_admin scoped to own tenant."""
    existing_key = await APIKeyService.get_by_id(db, key_id)
    if not existing_key:
        raise HTTPException(status_code=404, detail="API key not found")

    # Tenant scoping
    if not is_system_admin(admin.role):
        if existing_key.tenant_id != admin.tenant_id:
            raise HTTPException(status_code=403, detail="Cannot rename key in another tenant")

    api_key = await AdminAPIKeyService.rename_api_key(db, key_id, data.name)
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    await db.commit()
    return {"ok": True, "name": api_key.name}
