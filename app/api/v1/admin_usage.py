from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.core.auth_deps import require_admin
from app.db.models.user import User
from app.services.user_service import UserService
from app.services.admin_usage_service import AdminUsageService
from app.core.rbac import is_system_admin
from datetime import datetime, timezone

router = APIRouter(prefix="/api/v1/admin/usage", tags=["admin-usage"])


class UsageSummaryResponse(BaseModel):
    total_requests: int
    total_tokens: int
    total_storage: int
    top_users: list[dict]
    top_endpoints: list[dict]
    from_date: str
    to_date: str


class UserUsageResponse(BaseModel):
    daily_stats: list[dict]
    latest_logs: list[dict]


@router.get("/summary", response_model=UsageSummaryResponse)
async def get_usage_summary(
    from_date: str | None = Query(None, description="ISO format date"),
    to_date: str | None = Query(None, description="ISO format date"),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """
    Usage summary. system_admin sees all; tenant_admin sees own tenant only.
    """
    from_dt = None
    to_dt = None

    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
            # Safety: never pass naive datetimes into DB queries
            if from_dt.tzinfo is None:
                from_dt = from_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid from_date format")

    if to_date:
        try:
            to_dt = datetime.fromisoformat(to_date.replace("Z", "+00:00"))
            if to_dt.tzinfo is None:
                to_dt = to_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid to_date format")

    # Tenant scoping
    tenant_filter = None
    if not is_system_admin(admin.role):
        tenant_filter = admin.tenant_id

    summary = await AdminUsageService.get_usage_summary(
        db, from_dt, to_dt, tenant_id=tenant_filter
    )
    return summary


@router.get("/user/{user_id}", response_model=UserUsageResponse)
async def get_user_usage(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    user = await UserService.get_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Tenant scoping
    if not is_system_admin(admin.role):
        if user.tenant_id != admin.tenant_id:
            raise HTTPException(status_code=403, detail="Access denied: user is in another tenant")

    usage = await AdminUsageService.get_user_usage(db, user_id)

    # Serialize latest logs
    latest_logs = [
        {
            "id": log.id,
            "endpoint": log.endpoint,
            "method": log.method,
            "tokens_input": log.tokens_input,
            "tokens_output": log.tokens_output,
            "tokens_total": log.tokens_total,
            "file_size_bytes": log.file_size_bytes,
            "request_cost": float(log.request_cost),
            "status_code": log.status_code,
            "success": log.success,
            "created_at": log.created_at.isoformat(),
        }
        for log in usage["latest_logs"]
    ]

    return UserUsageResponse(
        daily_stats=usage["daily_stats"],
        latest_logs=latest_logs,
    )
