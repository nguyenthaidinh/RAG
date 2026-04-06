from fastapi import Request, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from jose import jwt, JWTError
from app.db.session import get_db
from app.services.user_service import UserService
from app.core.config import settings
from app.core.rbac import is_admin


async def get_admin_user_from_cookie(
    request: Request,
    db: AsyncSession,
) -> dict:
    """
    Get admin user from cookie JWT.
    Accepts system_admin, tenant_admin, and legacy 'admin' role.
    Returns user_dict.
    """
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALG])
        email = payload.get("sub")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    # Get user from DB
    user = await UserService.get_by_email(db, email)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    if not is_admin(user.role):
        raise HTTPException(status_code=403, detail="Admin access required")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="User account is inactive")

    user_dict = {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "tenant_id": user.tenant_id,
        "is_active": user.is_active,
    }

    return user_dict
