from fastapi import Depends, HTTPException, Request, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from jose import jwt, JWTError
from app.db.session import get_db
from app.services.user_service import UserService
from app.services.api_key_service import APIKeyService
from app.db.models.user import User
from app.core.config import settings
from app.core.rbac import ROLE_SYSTEM_ADMIN, ADMIN_ROLES


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
    access_token: str | None = Cookie(None),
) -> User:
    """
    Authenticate via JWT (Bearer or Cookie) OR API Key.
    Returns User object.
    Sets request.state.user_id, tenant_id, api_key_id for usage logging.
    """
    # Try API Key from X-API-Key header first
    api_key_header = request.headers.get("X-API-Key")
    if api_key_header:
        api_key = await APIKeyService.verify_api_key(db, api_key_header)
        if api_key:
            user = await UserService.get_by_id(db, api_key.user_id)
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            if not user.is_active:
                raise HTTPException(status_code=403, detail="User account is inactive")
            request.state.api_key_id = api_key.id
            request.state.user_id = user.id
            request.state.tenant_id = user.tenant_id
            return user

    # Try Authorization: Bearer header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()

        # Check if it's an API key: starts with "ak_" AND contains "."
        if token.startswith("ak_") and "." in token:
            api_key = await APIKeyService.verify_api_key(db, token)
            if api_key:
                user = await UserService.get_by_id(db, api_key.user_id)
                if not user:
                    raise HTTPException(status_code=401, detail="User not found")
                if not user.is_active:
                    raise HTTPException(status_code=403, detail="User account is inactive")
                request.state.api_key_id = api_key.id
                request.state.user_id = user.id
                request.state.tenant_id = user.tenant_id
                return user
        else:
            # Treat as JWT Bearer token
            user = await _authenticate_jwt(db, request, token)
            if user:
                return user

    # Try JWT from cookie (fallback for web)
    if access_token:
        user = await _authenticate_jwt(db, request, access_token)
        if user:
            return user

    raise HTTPException(status_code=401, detail="Authentication required")


async def _authenticate_jwt(
    db: AsyncSession,
    request: Request,
    token: str,
) -> User | None:
    """Decode JWT and return User if valid, else None."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
        email = payload.get("sub")
        if email:
            user = await UserService.get_by_email(db, email)
            if not user:
                return None
            if not user.is_active:
                return None
            request.state.user_id = user.id
            request.state.tenant_id = user.tenant_id
            return user
    except JWTError:
        pass
    return None


# ── Role-based dependencies ───────────────────────────────────────

async def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Require admin role (system_admin OR tenant_admin).
    """
    if current_user.role not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


async def require_system_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Require system_admin role.
    Only system_admin can create tenants, promote to system_admin, etc.
    """
    if current_user.role != ROLE_SYSTEM_ADMIN:
        raise HTTPException(status_code=403, detail="System admin access required")
    return current_user


async def require_tenant_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Require tenant_admin or system_admin role.
    tenant_admin is scoped to their own tenant (route handler must enforce).
    """
    if current_user.role not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="Tenant admin access required")
    return current_user
