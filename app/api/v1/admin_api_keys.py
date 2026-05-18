import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import require_admin
from app.core.rbac import is_system_admin
from app.db.models.api_key import APIKey
from app.db.models.user import User
from app.db.session import get_db
from app.services.api_key_service import APIKeyService
from app.services.user_service import UserService
from app.web.csrf import validate_csrf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin/api-keys", tags=["admin-api-keys"])


class CreateAPIKeyRequest(BaseModel):
    user_id: int
    name: str


class APIKeyResponse(BaseModel):
    id: int
    user_id: int
    tenant_id: str
    name: str
    prefix: str
    is_active: bool
    revoked_at: datetime | None
    last_used_at: datetime | None
    created_at: datetime


class CreateAPIKeyResponse(BaseModel):
    ok: bool
    api_key: APIKeyResponse
    plain_key: str


class RevokeAPIKeyResponse(BaseModel):
    ok: bool
    api_key: APIKeyResponse


class RotateAPIKeyResponse(BaseModel):
    ok: bool
    old_key: APIKeyResponse
    new_key: APIKeyResponse
    plain_key: str


def _serialize_api_key(api_key: APIKey) -> APIKeyResponse:
    return APIKeyResponse(
        id=api_key.id,
        user_id=api_key.user_id,
        tenant_id=api_key.tenant_id,
        name=api_key.name,
        prefix=api_key.prefix,
        is_active=api_key.is_active,
        revoked_at=api_key.revoked_at,
        last_used_at=api_key.last_used_at,
        created_at=api_key.created_at,
    )


def _hide_cross_tenant(actor: User, tenant_id: str) -> bool:
    return not is_system_admin(actor.role) and actor.tenant_id != tenant_id


def _ensure_mutable_api_key(api_key: APIKey) -> None:
    if not api_key.is_active or api_key.revoked_at is not None:
        raise HTTPException(status_code=409, detail="API key is already revoked")


def _validate_key_id(key_id: int) -> None:
    if key_id <= 0:
        raise HTTPException(status_code=400, detail="API key id must be positive")


async def require_cookie_csrf(request: Request) -> None:
    if request.cookies.get("access_token"):
        validate_csrf(request, request.headers.get("X-CSRF-Token"))


@router.post("/", response_model=CreateAPIKeyResponse)
async def create_api_key(
    data: CreateAPIKeyRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    _csrf: None = Depends(require_cookie_csrf),
) -> CreateAPIKeyResponse:
    name = data.name.strip()
    if data.user_id <= 0:
        raise HTTPException(status_code=400, detail="User id must be positive")
    if not name:
        raise HTTPException(status_code=400, detail="API key name is required")
    if len(name) > 255:
        raise HTTPException(status_code=400, detail="API key name is too long")

    try:
        target_user = await UserService.get_by_id(db, data.user_id)
        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")
        if _hide_cross_tenant(admin, target_user.tenant_id):
            raise HTTPException(status_code=404, detail="User not found")

        api_key, plain_key = await APIKeyService.create_api_key(
            db,
            user_id=target_user.id,
            tenant_id=target_user.tenant_id,
            name=name,
            actor_user_id=admin.id,
        )
        await db.commit()
    except HTTPException:
        raise
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        await db.rollback()
        logger.exception("admin_api_key.create_failed")
        raise HTTPException(status_code=500, detail="Failed to create API key")

    return CreateAPIKeyResponse(
        ok=True,
        api_key=_serialize_api_key(api_key),
        plain_key=plain_key,
    )


@router.post("/{key_id}/revoke", response_model=RevokeAPIKeyResponse)
async def revoke_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    _csrf: None = Depends(require_cookie_csrf),
) -> RevokeAPIKeyResponse:
    _validate_key_id(key_id)

    try:
        api_key = await APIKeyService.get_by_id(db, key_id)
        if not api_key:
            raise HTTPException(status_code=404, detail="API key not found")
        if _hide_cross_tenant(admin, api_key.tenant_id):
            raise HTTPException(status_code=404, detail="API key not found")
        _ensure_mutable_api_key(api_key)

        revoked_key = await APIKeyService.revoke_api_key(
            db,
            key_id,
            actor_user_id=admin.id,
        )
        if not revoked_key:
            raise HTTPException(status_code=404, detail="API key not found")
        await db.commit()
    except HTTPException:
        raise
    except Exception:
        await db.rollback()
        logger.exception("admin_api_key.revoke_failed")
        raise HTTPException(status_code=500, detail="Failed to revoke API key")

    return RevokeAPIKeyResponse(
        ok=True,
        api_key=_serialize_api_key(revoked_key),
    )


@router.post("/{key_id}/rotate", response_model=RotateAPIKeyResponse)
async def rotate_api_key(
    key_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
    _csrf: None = Depends(require_cookie_csrf),
) -> RotateAPIKeyResponse:
    _validate_key_id(key_id)

    try:
        old_key = await APIKeyService.get_by_id(db, key_id)
        if not old_key:
            raise HTTPException(status_code=404, detail="API key not found")
        if _hide_cross_tenant(admin, old_key.tenant_id):
            raise HTTPException(status_code=404, detail="API key not found")
        _ensure_mutable_api_key(old_key)

        result = await APIKeyService.rotate_api_key(
            db,
            key_id,
            actor_user_id=admin.id,
        )
        if not result:
            raise HTTPException(status_code=404, detail="API key not found")

        new_key, plain_key = result
        await db.commit()
    except HTTPException:
        raise
    except ValueError as exc:
        await db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception:
        await db.rollback()
        logger.exception("admin_api_key.rotate_failed")
        raise HTTPException(status_code=500, detail="Failed to rotate API key")

    return RotateAPIKeyResponse(
        ok=True,
        old_key=_serialize_api_key(old_key),
        new_key=_serialize_api_key(new_key),
        plain_key=plain_key,
    )
