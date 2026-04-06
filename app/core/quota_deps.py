from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.auth_deps import get_current_user
from app.db.models.user import User
from app.services.quota_service import QuotaService


async def enforce_quota(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Dependency to enforce quota limits at tenant level (atomic).
    Raises 429 if quota exceeded, 403 if inactive.

    Falls back to user-level quota if no tenant quota exists.
    """
    allowed, error_msg = await QuotaService.check_and_enforce_quota_by_user(
        db, current_user.id, current_user.tenant_id
    )
    if not allowed:
        if error_msg and "inactive" in error_msg.lower():
            raise HTTPException(status_code=403, detail=error_msg)
        raise HTTPException(status_code=429, detail=error_msg or "Quota exceeded")
    return current_user
