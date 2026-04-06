from fastapi import APIRouter, Depends
from app.core.auth_deps import get_current_user
from app.db.models.user import User

router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {
        "email": user.email,
        "role": user.role,
        "tenant_id": user.tenant_id,
        "is_active": user.is_active,
    }
