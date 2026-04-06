from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.auth_deps import require_admin, require_system_admin
from app.db.models.user import User
from app.services.user_service import UserService
from app.services.quota_service import QuotaService
from app.services.admin_user_service import AdminUserService
from app.core.rbac import VALID_ROLES, is_system_admin
from datetime import datetime

router = APIRouter(prefix="/api/v1/admin/users", tags=["admin-users"])


class CreateUserRequest(BaseModel):
    email: EmailStr
    password: str
    tenant_id: str
    role: str = "user"
    is_active: bool = True


class UserResponse(BaseModel):
    id: int
    email: str
    role: str
    tenant_id: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    items: list[UserResponse]
    total: int
    page: int
    page_size: int


class UserDetailResponse(UserResponse):
    quota: dict
    api_keys_count: int
    recent_usage: list[dict]


class UpdateStatusRequest(BaseModel):
    is_active: bool


class UpdateRoleRequest(BaseModel):
    role: str


@router.get("/", response_model=UserListResponse)
async def list_users(
    q: str | None = Query(None, description="Search email"),
    status: str = Query("all", description="active|inactive|all"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    List users. system_admin sees all; tenant_admin sees own tenant only.
    """
    users, total = await AdminUserService.search_users(
        db, actor=admin, q=q, status=status, page=page, page_size=page_size
    )
    return UserListResponse(
        items=users,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/", response_model=UserResponse)
async def create_user(
    data: CreateUserRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Create user. RBAC enforced in service layer:
    - system_admin: any tenant, any role
    - tenant_admin: own tenant only, cannot assign system_admin
    """
    try:
        user = await AdminUserService.create_user(
            db,
            email=data.email,
            password=data.password,
            tenant_id=data.tenant_id,
            role=data.role,
            is_active=data.is_active,
            actor=admin,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    await db.commit()
    return user


@router.patch("/{user_id}/status")
async def set_user_status(
    user_id: int,
    data: UpdateStatusRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Toggle user active status. Tenant scoping enforced in service."""
    user = await AdminUserService.toggle_user_status(
        db, user_id, data.is_active, actor=admin
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await db.commit()
    return {"ok": True, "is_active": user.is_active}


@router.patch("/{user_id}/role")
async def set_user_role(
    user_id: int,
    data: UpdateRoleRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Change user role. RBAC enforced in service layer."""
    if data.role not in VALID_ROLES:
        raise HTTPException(
            status_code=422,
            detail=f"Role must be one of: {', '.join(sorted(VALID_ROLES))}",
        )

    try:
        user = await AdminUserService.update_user_role(
            db, user_id, data.role, actor=admin
        )
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    await db.commit()
    return {"ok": True, "role": user.role}


@router.get("/{user_id}", response_model=UserDetailResponse)
async def get_user_detail(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Get user detail. Tenant scoping enforced in service."""
    detail = await AdminUserService.get_user_detail(db, user_id, actor=admin)
    if not detail:
        raise HTTPException(status_code=404, detail="User not found")

    user = detail["user"]
    quota = detail["quota"]
    api_keys_count = detail["api_keys_count"]
    recent_usage = detail["recent_usage"]

    quota_dict = {}
    if quota:
        quota_dict = {
            "plan": quota.plan,
            "is_active": quota.is_active,
            "max_tokens": quota.max_tokens,
            "used_tokens": quota.used_tokens,
            "max_requests": quota.max_requests,
            "used_requests": quota.used_requests,
            "max_storage_mb": quota.max_storage_mb,
            "used_storage_mb": quota.used_storage_mb,
        }

    usage_list = [
        {
            "id": u.id,
            "endpoint": u.endpoint,
            "method": u.method,
            "tokens_total": u.tokens_total,
            "status_code": u.status_code,
            "success": u.success,
            "created_at": u.created_at.isoformat(),
        }
        for u in recent_usage
    ]

    return UserDetailResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        tenant_id=user.tenant_id,
        is_active=user.is_active,
        created_at=user.created_at,
        quota=quota_dict,
        api_keys_count=api_keys_count,
        recent_usage=usage_list,
    )
