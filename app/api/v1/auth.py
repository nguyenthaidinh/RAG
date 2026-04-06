import hashlib
import logging

from fastapi import APIRouter, Request, Response, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.services.auth_service import create_access_token
from app.core.security import verify_password
from app.core.config import settings
from app.db.session import get_db
from app.db.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def _hash_for_audit(value: str) -> str:
    """SHA-256 truncated to 12 hex chars — privacy-safe identifier."""
    return hashlib.sha256(value.encode()).hexdigest()[:12]


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/login")
async def login(
    data: LoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_db),
):
    result = await session.execute(
        select(User).where(User.email == data.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(data.password, user.password_hash):
        # ── Phase 6: emit LOGIN_FAILURE audit (fail-open) ──
        try:
            from app.services.audit_service import get_audit_service
            await get_audit_service().log_login_failure(
                session,
                tenant_id=user.tenant_id if user else "_unknown",
                request_id=getattr(request.state, "request_id", None),
                email_hash=_hash_for_audit(data.email),
                login_method="api",
            )
        except Exception:
            logger.warning("audit.login_failure_failed", exc_info=True)

        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user.email)

    # ── Phase 6: emit LOGIN_SUCCESS audit (fail-open) ──
    try:
        from app.services.audit_service import get_audit_service
        await get_audit_service().log_login_success(
            session,
            tenant_id=user.tenant_id,
            user_id=user.id,
            request_id=getattr(request.state, "request_id", None),
            email_hash=_hash_for_audit(data.email),
            login_method="api",
        )
    except Exception:
        logger.warning("audit.login_success_failed", exc_info=True)

    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_SECONDS,
    }


@router.post("/login-web")
async def login_web(
    data: LoginRequest,
    request: Request,
    response: Response,
    session: AsyncSession = Depends(get_db),
):
    result = await session.execute(
        select(User).where(User.email == data.email)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(data.password, user.password_hash):
        # ── Phase 6: emit LOGIN_FAILURE audit (fail-open) ──
        try:
            from app.services.audit_service import get_audit_service
            await get_audit_service().log_login_failure(
                session,
                tenant_id=user.tenant_id if user else "_unknown",
                request_id=getattr(request.state, "request_id", None),
                email_hash=_hash_for_audit(data.email),
                login_method="web",
            )
        except Exception:
            logger.warning("audit.login_failure_failed", exc_info=True)

        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user.email)

    # ── Phase 6: emit LOGIN_SUCCESS audit (fail-open) ──
    try:
        from app.services.audit_service import get_audit_service
        await get_audit_service().log_login_success(
            session,
            tenant_id=user.tenant_id,
            user_id=user.id,
            request_id=getattr(request.state, "request_id", None),
            email_hash=_hash_for_audit(data.email),
            login_method="web",
        )
    except Exception:
        logger.warning("audit.login_success_failed", exc_info=True)

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=settings.ENV == "prod",
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_SECONDS,
        path="/",
    )

    return {"ok": True}
