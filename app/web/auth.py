from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.services.user_service import UserService


def _is_secure_cookie() -> bool:
    """
    Return True when cookies must carry the Secure flag (HTTPS only).

    Matches both the Settings default ``ENV="prod"`` **and** the value
    ``ENV=production`` found in the project ``.env`` file.  Any other
    value (dev, local, staging, test …) → False → cookies work over
    plain HTTP during development.
    """
    return settings.ENV.lower() in ("prod", "production")


# =========================
# VERIFY COOKIE TOKEN
# =========================
async def get_current_user_from_cookie(
    request: Request,
    db: AsyncSession,
) -> dict:
    """
    Decode the JWT stored in the ``access_token`` HttpOnly cookie and
    return a lightweight user dict.

    JWT contract (see ``app/services/auth_service.create_access_token``):
      • ``sub`` = **user.email**  (passed as ``subject`` argument)
      • ``iat`` / ``exp`` = standard issued-at / expiry

    Decode logic mirrors ``app/web/admin_helpers.get_admin_user_from_cookie``
    and ``app/core/auth_deps._authenticate_jwt`` — same secret, same
    algorithm, same claim → email → DB lookup path.  No admin-role gate.
    """
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401)

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALG],
        )
    except JWTError:
        raise HTTPException(status_code=401)

    # ``sub`` holds the user's email — established by
    # auth_service.create_access_token(user.email).
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401)

    user = await UserService.get_by_email(db, sub)
    if not user:
        raise HTTPException(status_code=401)

    if not user.is_active:
        raise HTTPException(status_code=401)

    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "tenant_id": user.tenant_id,
        "is_active": user.is_active,
    }


# =========================
# LOGIN VIA API
# =========================
async def login_via_api(email: str, password: str):
    import httpx

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "http://127.0.0.1:8000/api/v1/auth/login",
            json={
                "email": email,
                "password": password,
            },
            timeout=5.0,
        )

    if r.status_code != 200:
        return None

    return r.json()


# =========================
# LOGIN RESPONSE (set cookies)
# =========================
def create_login_response(access_token: str) -> RedirectResponse:
    import secrets

    response = RedirectResponse("/dashboard", status_code=302)
    secure = _is_secure_cookie()

    # ── Auth cookie: HttpOnly, SameSite=Lax, path=/ ──
    # NEVER expose token to JS.  No domain= (exact origin match).
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
        max_age=60 * 60 * 8,
    )

    # ── CSRF double-submit cookie (readable by JS, NOT HttpOnly) ──
    # This cookie is part of the existing CSRF protection implemented
    # in app/web/csrf.py.  JS reads the value and sends it back as a
    # hidden form field; the server compares cookie vs field on POST.
    csrf_token = secrets.token_urlsafe(32)
    response.set_cookie(
        key="csrf_token",
        value=csrf_token,
        httponly=False,
        secure=secure,
        samesite="lax",
        path="/",
        max_age=60 * 60 * 8,
    )

    return response
