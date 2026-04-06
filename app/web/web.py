from fastapi import APIRouter, Request, Depends, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.web.auth import (
    get_current_user_from_cookie,
    login_via_api,
    create_login_response,
    _is_secure_cookie,
)
from app.web.admin_helpers import get_admin_user_from_cookie
from app.web.csrf import validate_csrf
from app.web.i18n import get_translator, DEFAULT_LANG, SUPPORTED_LANGS
from app.services.admin_user_service import AdminUserService
from app.services.admin_api_key_service import AdminAPIKeyService
from app.services.admin_usage_service import AdminUsageService
from app.services.query_analytics_service import QueryAnalyticsService
from app.services.quota_service import QuotaService
from app.services.api_key_service import APIKeyService
from app.services.quota_policy_service import QuotaPolicyService
from app.services.user_service import UserService
from app.repos.plan_repo import PlanRepository
from app.repos.tenant_setting_repo import TenantSettingRepository
from app.db.session import get_db
from app.core.rbac import is_admin, is_system_admin

router = APIRouter()


# ── i18n: language-aware template engine ───────────────────────────────


def _get_lang(request: Request) -> str:
    """Read language preference from cookie, default to vi."""
    lang = request.cookies.get("lang", DEFAULT_LANG)
    return lang if lang in SUPPORTED_LANGS else DEFAULT_LANG


class I18nTemplates(Jinja2Templates):
    """
    Auto-inject ``t`` (translator) and ``lang`` into every template context.

    This means ALL routes automatically get i18n support without needing
    to manually pass ``t`` and ``lang`` in every TemplateResponse call.
    """

    def TemplateResponse(
        self,
        name: str,
        context: dict,
        **kwargs,
    ):
        request = context.get("request")
        if request and "t" not in context:
            lang = _get_lang(request)
            context["t"] = get_translator(lang)
            context["lang"] = lang
        return super().TemplateResponse(name, context, **kwargs)


templates = I18nTemplates(directory="app/templates")
templates.env.globals["SUPPORTED_LANGS"] = SUPPORTED_LANGS


# ── Static file cache-busting ─────────────────────────────────────────
# Generates /static/css/foo.css?h=<md5_8chars> so browsers/CDNs
# automatically fetch the new version whenever file content changes.

import hashlib
from functools import lru_cache
from pathlib import Path as _Path

_STATIC_ROOT = _Path(__file__).resolve().parent.parent / "static"


@lru_cache(maxsize=128)
def _file_hash(rel_path: str) -> str:
    """Return first 8 hex chars of MD5 for a static file."""
    fp = _STATIC_ROOT / rel_path
    if fp.is_file():
        return hashlib.md5(fp.read_bytes()).hexdigest()[:8]
    return "0"


def static_url(rel_path: str) -> str:
    """Build a cache-busted URL for a static asset.

    Usage in Jinja2: {{ static_url('css/dashboard.css') }}
    Output:          /static/css/dashboard.css?h=a1b2c3d4
    """
    h = _file_hash(rel_path)
    return f"/static/{rel_path}?h={h}"


templates.env.globals["static_url"] = static_url


# =========================
# HOME
# =========================
@router.get("/", response_class=HTMLResponse)
async def home():
    return RedirectResponse("/dashboard")


# =========================
# LOGIN
# =========================
@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(
        "login.html",
        {"request": request},
    )


@router.post("/login", response_class=HTMLResponse)
async def login_action(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
):
    result = await login_via_api(email, password)

    if not result:
        t = get_translator(_get_lang(request))
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": t("login_error"),
            },
            status_code=401,
        )

    return create_login_response(result["access_token"])


# =========================
# DASHBOARD HOME
# =========================
@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        user = await get_current_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    # Non-admin users get a minimal user dashboard (no admin sidebar)
    if not is_admin(user.get("role", "")):
        return templates.TemplateResponse(
            "user_dashboard.html",
            {
                "request": request,
                "user": user,
            },
        )

    # Security: NEVER read access_token from cookie into template context.
    # Auth relies solely on HttpOnly cookie — no token exposure to Jinja/JS.

    # Query analytics KPIs
    qa_svc = QueryAnalyticsService()
    tenant_scope = None if is_system_admin(user.get("role", "")) else user.get("tenant_id")
    try:
        query_kpis = await qa_svc.get_dashboard_kpis(db, tenant_id=tenant_scope)
    except Exception:
        query_kpis = None

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "query_kpis": query_kpis,
        },
    )


# =========================
# API TEST
# =========================
@router.get("/dashboard/api-test", response_class=HTMLResponse)
async def api_test(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        user = await get_current_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_admin(user.get("role", "")):
        raise HTTPException(status_code=403)

    return templates.TemplateResponse(
        "api_test.html",
        {
            "request": request,
            "user": user,
        },
    )


# =========================
# USERS LIST
# =========================
@router.get("/dashboard/users", response_class=HTMLResponse)
async def dashboard_users(
    request: Request,
    q: str | None = None,
    status: str = "all",
    page: int = 1,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    # Load actor for tenant-scoped service calls
    actor = await UserService.get_by_id(db, admin["id"])

    users, total = await AdminUserService.search_users(
        db,
        actor=actor,
        q=q,
        status=status,
        page=page,
        page_size=50,
    )

    return templates.TemplateResponse(
        "dashboard/users.html",
        {
            "request": request,
            "user": admin,
            "users": users,
            "total": total,
            "page": page,
            "q": q or "",
            "status": status,
        },
    )


# =========================
# NEW USER (FORM)
# =========================
@router.get("/dashboard/users/new", response_class=HTMLResponse)
async def dashboard_user_new(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    return templates.TemplateResponse(
        "dashboard/user_new.html",
        {
            "request": request,
            "user": admin,
        },
    )


# =========================
# CREATE USER (POST) — CSRF-protected
# =========================
@router.post("/dashboard/users/new", response_class=HTMLResponse)
async def dashboard_user_create(
    request: Request,
    email: str = Form(""),
    password: str = Form(""),
    tenant_id: str = Form(""),
    role: str = Form("user"),
    is_active: str = Form("true"),
    csrf_token: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    # CSRF double-submit validation (cookie vs form field)
    validate_csrf(request, csrf_token)

    # Validate required fields
    if not email.strip():
        return templates.TemplateResponse(
            "dashboard/user_new.html",
            {"request": request, "user": admin, "error": "Email is required"},
            status_code=400,
        )
    if not password:
        return templates.TemplateResponse(
            "dashboard/user_new.html",
            {"request": request, "user": admin, "error": "Password is required"},
            status_code=400,
        )

    # Resolve tenant_id: tenant_admin is locked to own tenant
    resolved_tenant_id = tenant_id.strip()
    if not is_system_admin(admin.get("role", "")):
        resolved_tenant_id = admin.get("tenant_id", "")

    if not resolved_tenant_id:
        return templates.TemplateResponse(
            "dashboard/user_new.html",
            {"request": request, "user": admin, "error": "Tenant ID is required"},
            status_code=400,
        )

    # Parse is_active to boolean — accept "true", "false", "1", "0"
    is_active_bool = str(is_active).lower() in ("true", "1", "yes")

    # Load actor for RBAC enforcement
    actor = await UserService.get_by_id(db, admin["id"])

    try:
        await AdminUserService.create_user(
            db,
            email=email.strip().lower(),
            password=password,
            tenant_id=resolved_tenant_id,
            role=role,
            is_active=is_active_bool,
            actor=actor,
        )

        await db.commit()

        return RedirectResponse(
            "/dashboard/users",
            status_code=303,
        )

    except ValueError as e:
        await db.rollback()
        return templates.TemplateResponse(
            "dashboard/user_new.html",
            {
                "request": request,
                "user": admin,
                "error": str(e),
            },
            status_code=400,
        )

    except Exception:
        await db.rollback()
        return templates.TemplateResponse(
            "dashboard/user_new.html",
            {
                "request": request,
                "user": admin,
                "error": "An unexpected error occurred. Please try again.",
            },
            status_code=500,
        )


# =========================
# USER DETAIL
# =========================
@router.get("/dashboard/users/{user_id}", response_class=HTMLResponse)
async def dashboard_user_detail(
    request: Request,
    user_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    # Load actor for tenant-scoped service calls
    actor = await UserService.get_by_id(db, admin["id"])

    # get_user_detail returns None for cross-tenant access (404, not 403)
    detail = await AdminUserService.get_user_detail(db, user_id, actor=actor)
    if not detail:
        raise HTTPException(status_code=404, detail="User not found")

    quota = await QuotaService.get_by_user_id(db, user_id)
    api_keys = await APIKeyService.list_by_user_id(db, user_id)

    # Query analytics for this user (Phase 4.2)
    qa_svc = QueryAnalyticsService()
    try:
        from datetime import datetime as _dt, timedelta, timezone as _tz
        _now = _dt.now(_tz.utc)
        user_query_stats = await qa_svc._repo.summary(
            db,
            from_dt=_now - timedelta(days=7),
            to_dt=_now,
            user_id=user_id,
        )
    except Exception:
        user_query_stats = None

    return templates.TemplateResponse(
        "dashboard/user_detail.html",
        {
            "request": request,
            "user": admin,
            "target_user": detail["user"],
            "quota": quota,
            "api_keys": api_keys,
            "recent_usage": detail["recent_usage"][:20],
            "user_query_stats": user_query_stats,
        },
    )


# =========================
# API KEYS
# =========================
@router.get("/dashboard/api-keys", response_class=HTMLResponse)
async def dashboard_api_keys(
    request: Request,
    user_id: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    keys = await AdminAPIKeyService.list_api_keys(db, user_id=user_id)

    return templates.TemplateResponse(
        "dashboard/api_keys.html",
        {
            "request": request,
            "user": admin,
            "api_keys": keys,
            "filter_user_id": user_id,
        },
    )


# =========================
# USAGE
# =========================
@router.get("/dashboard/usage", response_class=HTMLResponse)
async def dashboard_usage(
    request: Request,
    from_date: str | None = None,
    to_date: str | None = None,
    mode: str | None = None,
    tenant_id: str | None = None,
    user_id: int | None = None,
    page: int = 1,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    summary = await AdminUsageService.get_usage_summary(db)

    # ── Query usage (Phase 4.2) ──
    from datetime import datetime as _dt, timezone as _tz

    qa_svc = QueryAnalyticsService()
    qa_from = None
    qa_to = None
    if from_date:
        try:
            qa_from = _dt.fromisoformat(from_date)
            # Security: never pass naive datetimes into DB queries
            if qa_from.tzinfo is None:
                qa_from = qa_from.replace(tzinfo=_tz.utc)
        except ValueError:
            pass
    if to_date:
        try:
            qa_to = _dt.fromisoformat(to_date)
            if qa_to.tzinfo is None:
                qa_to = qa_to.replace(tzinfo=_tz.utc)
        except ValueError:
            pass

    # Tenant scoping
    qa_tenant = tenant_id
    if not is_system_admin(admin.get("role", "")):
        qa_tenant = admin.get("tenant_id")

    page_size = 50
    offset = (page - 1) * page_size

    try:
        query_page = await qa_svc.get_query_usage_page(
            db,
            from_dt=qa_from,
            to_dt=qa_to,
            tenant_id=qa_tenant,
            user_id=user_id,
            mode=mode if mode else None,
            limit=page_size,
            offset=offset,
        )
    except Exception:
        query_page = {"items": [], "total": 0, "summary": {"total_queries": 0, "total_tokens": 0, "avg_latency_ms": 0}}

    import math
    total_pages = max(1, math.ceil(query_page["total"] / page_size))

    return templates.TemplateResponse(
        "dashboard/usage.html",
        {
            "request": request,
            "user": admin,
            "summary": summary,
            # Phase 4.2 query usage data
            "query_items": query_page["items"],
            "query_total": query_page["total"],
            "query_summary": query_page["summary"],
            "query_page": page,
            "query_total_pages": total_pages,
            # Preserve filter state
            "filter_from": from_date or "",
            "filter_to": to_date or "",
            "filter_mode": mode or "",
            "filter_tenant": tenant_id or "",
            "filter_user": user_id or "",
        },
    )


# =========================
# AUDIT LOG — Phase 6.0
# =========================
@router.get("/dashboard/audit", response_class=HTMLResponse)
async def dashboard_audit(
    request: Request,
    page: int = 1,
    db: AsyncSession = Depends(get_db),
    # Filter params (match API query params)
    from_dt: str | None = None,
    to_dt: str | None = None,
    tenant_id: str | None = None,
    event_type: str | None = None,
    severity: str | None = None,
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    from datetime import datetime as _dt, timedelta, timezone as _tz
    from app.repos.audit_event_repo import AuditEventFilters, AuditEventRepository
    from app.core.config import settings as app_settings

    now = _dt.now(_tz.utc)

    # Parse date filters
    parsed_from = None
    parsed_to = None
    if from_dt:
        try:
            parsed_from = _dt.fromisoformat(from_dt)
            if parsed_from.tzinfo is None:
                parsed_from = parsed_from.replace(tzinfo=_tz.utc)
        except ValueError:
            pass
    if to_dt:
        try:
            parsed_to = _dt.fromisoformat(to_dt)
            if parsed_to.tzinfo is None:
                parsed_to = parsed_to.replace(tzinfo=_tz.utc)
        except ValueError:
            pass

    # Defaults: last N days
    if parsed_from is None and parsed_to is None:
        parsed_from = now - timedelta(days=app_settings.AUDIT_DEFAULT_WINDOW_DAYS)
        parsed_to = now

    # Cap max window
    if parsed_from and parsed_to:
        max_delta = timedelta(days=app_settings.AUDIT_MAX_WINDOW_DAYS)
        if (parsed_to - parsed_from) > max_delta:
            parsed_from = parsed_to - max_delta

    # Tenant scoping
    scoped_tenant = None
    _is_sys_admin = is_system_admin(admin.get("role", ""))
    if _is_sys_admin:
        scoped_tenant = tenant_id  # may be None (all tenants)
    else:
        scoped_tenant = admin.get("tenant_id")

    page_size = 50
    offset = (page - 1) * page_size

    repo = AuditEventRepository()
    filters = AuditEventFilters(
        from_dt=parsed_from,
        to_dt=parsed_to,
        tenant_id=scoped_tenant,
        event_type=event_type,
        severity=severity,
    )
    items, total = await repo.list_events(db, filters, limit=page_size, offset=offset)

    import math
    total_pages = max(1, math.ceil(total / page_size))

    return templates.TemplateResponse(
        "dashboard/audit.html",
        {
            "request": request,
            "user": admin,
            "items": items,
            "total": total,
            "page": page,
            "total_pages": total_pages,
            "is_system_admin": _is_sys_admin,
            # Preserve filter state
            "filter_from": from_dt or "",
            "filter_to": to_dt or "",
            "filter_tenant": tenant_id or "",
            "filter_event_type": event_type or "",
            "filter_severity": severity or "",
        },
    )


# =========================
# TENANTS LIST (system_admin only)
# =========================
@router.get("/dashboard/tenants", response_class=HTMLResponse)
async def dashboard_tenants(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_system_admin(admin.get("role", "")):
        raise HTTPException(status_code=403, detail="System admin access required")

    from app.db.models.tenant import Tenant
    from sqlalchemy import select as sa_select

    res = await db.execute(sa_select(Tenant).order_by(Tenant.created_at.desc()))
    tenants = res.scalars().all()

    return templates.TemplateResponse(
        "dashboard/tenants.html",
        {
            "request": request,
            "user": admin,
            "tenants": tenants,
        },
    )


# =========================
# NEW TENANT (FORM)
# =========================
@router.get("/dashboard/tenants/new", response_class=HTMLResponse)
async def dashboard_tenant_new(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_system_admin(admin.get("role", "")):
        raise HTTPException(status_code=403)

    return templates.TemplateResponse(
        "dashboard/tenant_new.html",
        {
            "request": request,
            "user": admin,
            "error": None,
            "form_data": None,
        },
    )


# =========================
# CREATE TENANT (POST) — CSRF-protected
# =========================
@router.post("/dashboard/tenants/new", response_class=HTMLResponse)
async def dashboard_tenant_create(
    request: Request,
    tenant_id: str = Form(""),
    name: str = Form(""),
    max_users: int = Form(10),
    max_requests: int = Form(10_000),
    max_tokens: int = Form(10_000_000),
    max_storage_mb: int = Form(1024),
    csrf_token: str = Form(""),
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_system_admin(admin.get("role", "")):
        raise HTTPException(status_code=403)

    validate_csrf(request, csrf_token)

    form_data = {
        "tenant_id": tenant_id,
        "name": name,
        "max_users": max_users,
        "max_requests": max_requests,
        "max_tokens": max_tokens,
        "max_storage_mb": max_storage_mb,
    }

    # Validate
    tid = tenant_id.strip().lower()
    if not tid or len(tid) < 3:
        return templates.TemplateResponse(
            "dashboard/tenant_new.html",
            {"request": request, "user": admin, "error": "Tenant ID must be at least 3 characters", "form_data": form_data},
            status_code=400,
        )

    if not name.strip():
        return templates.TemplateResponse(
            "dashboard/tenant_new.html",
            {"request": request, "user": admin, "error": "Tenant name is required", "form_data": form_data},
            status_code=400,
        )

    from app.db.models.tenant import Tenant
    from sqlalchemy import select as sa_select

    # Check duplicate
    existing = await db.execute(sa_select(Tenant).where(Tenant.id == tid))
    if existing.scalar_one_or_none():
        return templates.TemplateResponse(
            "dashboard/tenant_new.html",
            {"request": request, "user": admin, "error": f"Tenant '{tid}' already exists", "form_data": form_data},
            status_code=409,
        )

    try:
        tenant = Tenant(
            id=tid,
            name=name.strip(),
            is_active=True,
            max_users=max_users,
        )
        db.add(tenant)
        await db.flush()

        # Create tenant quota
        await QuotaService.create_tenant_quota(
            db,
            tenant_id=tid,
            max_requests=max_requests,
            max_tokens=max_tokens,
            max_storage_mb=max_storage_mb,
        )

        await db.commit()

        return RedirectResponse("/dashboard/tenants", status_code=303)

    except Exception as e:
        await db.rollback()
        return templates.TemplateResponse(
            "dashboard/tenant_new.html",
            {"request": request, "user": admin, "error": f"Failed to create tenant: {str(e)[:200]}", "form_data": form_data},
            status_code=500,
        )


# =========================
# TENANT DETAIL
# =========================
@router.get("/dashboard/tenants/{tenant_id}", response_class=HTMLResponse)
async def dashboard_tenant_detail(
    request: Request,
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_system_admin(admin.get("role", "")):
        raise HTTPException(status_code=403)

    from app.db.models.tenant import Tenant
    from app.db.models.user import User
    from sqlalchemy import select as sa_select, func

    tenant = (
        await db.execute(sa_select(Tenant).where(Tenant.id == tenant_id))
    ).scalar_one_or_none()

    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Get users in this tenant
    users_res = await db.execute(
        sa_select(User)
        .where(User.tenant_id == tenant_id)
        .order_by(User.created_at.desc())
    )
    users = users_res.scalars().all()

    user_count = len(users)

    return templates.TemplateResponse(
        "dashboard/tenant_detail.html",
        {
            "request": request,
            "user": admin,
            "tenant": tenant,
            "users": users,
            "user_count": user_count,
        },
    )


# =========================
# DOCUMENTS — Phase 2A
# =========================
@router.get("/dashboard/documents", response_class=HTMLResponse)
async def dashboard_documents(
    request: Request,
    q: str | None = None,
    status: str | None = None,
    representation_type: str | None = None,
    source: str | None = None,
    tenant_id: str | None = None,
    page: int = 1,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_admin(admin.get("role", "")):
        raise HTTPException(status_code=403)

    from app.services.document_admin_service import DocumentAdminService
    from app.repos.document_admin_repo import DocumentAdminRepo

    svc = DocumentAdminService()
    repo = DocumentAdminRepo()

    _is_sys_admin = is_system_admin(admin.get("role", ""))
    scoped_tenant = (
        tenant_id if _is_sys_admin and tenant_id
        else admin.get("tenant_id")
    )

    page_size = 50
    docs, total = await svc.get_list(
        db,
        tenant_id=scoped_tenant,
        q=q,
        status=status,
        representation_type=representation_type,
        source=source,
        page=page,
        page_size=page_size,
    )

    sources = await repo.get_distinct_sources(db, tenant_id=scoped_tenant)

    import math
    total_pages = max(1, math.ceil(total / page_size))

    return templates.TemplateResponse(
        "dashboard/documents.html",
        {
            "request": request,
            "user": admin,
            "documents": docs,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "q": q or "",
            "filter_status": status or "",
            "filter_repr": representation_type or "",
            "filter_source": source or "",
            "filter_tenant": tenant_id or "",
            "is_system_admin": _is_sys_admin,
            "sources": sources,
        },
    )


@router.get("/dashboard/documents/{document_id}", response_class=HTMLResponse)
async def dashboard_document_detail(
    request: Request,
    document_id: int,
    tenant_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_admin(admin.get("role", "")):
        raise HTTPException(status_code=403)

    from app.services.document_admin_service import DocumentAdminService
    from app.repos.document_admin_repo import DocumentAdminRepo

    # Fix #1: system_admin can view cross-tenant documents
    _is_sys_admin = is_system_admin(admin.get("role", ""))
    scoped_tenant = (
        tenant_id if _is_sys_admin and tenant_id
        else admin.get("tenant_id")
    )

    svc = DocumentAdminService()
    repo = DocumentAdminRepo()

    doc = await repo.get_document_by_id(
        db, document_id=document_id, tenant_id=scoped_tenant,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    detail = await svc.get_detail(
        db, document_id=document_id, tenant_id=scoped_tenant,
    )

    events = await repo.list_document_events(
        db, document_id=document_id, tenant_id=scoped_tenant,
    )

    # Fix #3: Use service for preview — single source of truth for truncation
    preview_raw = await svc.get_open_payload(
        db, document_id=document_id, tenant_id=scoped_tenant, view="raw",
    )
    preview_cleaned = await svc.get_open_payload(
        db, document_id=document_id, tenant_id=scoped_tenant, view="cleaned",
    )
    preview_synthesized = None
    if doc.representation_type == "synthesized":
        preview_synthesized = await svc.get_open_payload(
            db, document_id=document_id, tenant_id=scoped_tenant, view="synthesized",
        )

    # Effective tenant for template links (preserve for system_admin cross-tenant)
    effective_tenant = scoped_tenant if _is_sys_admin and tenant_id else ""

    return templates.TemplateResponse(
        "dashboard/document_detail.html",
        {
            "request": request,
            "user": admin,
            "doc": doc,
            "metadata": detail["metadata"] if detail else {},
            "content_stats": detail["content_stats"] if detail else {},
            "related": detail["related"] if detail else {"parent": None, "children": []},
            "events": events,
            "preview_raw": preview_raw,
            "preview_cleaned": preview_cleaned,
            "preview_synthesized": preview_synthesized,
            "is_system_admin": _is_sys_admin,
            "tenant_id_param": effective_tenant,
            # Phase 2B: action button preconditions
            "can_retry_ingest": doc.status in ("error", "uploaded", "chunked", "indexed") and bool(doc.content_raw),
            "can_reindex": doc.status in ("chunked", "indexed", "ready") and bool(doc.content_text or doc.content_raw),
            "can_resynthesize": doc.representation_type == "original" and bool(doc.content_text or doc.content_raw),
            "flash_message": request.query_params.get("msg", ""),
            "flash_ok": request.query_params.get("ok", ""),
        },
    )


# =========================
# DOCUMENT ACTIONS — Phase 2B
# =========================
@router.post("/dashboard/documents/{document_id}/retry-ingest")
async def dashboard_retry_ingest(
    request: Request,
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_admin(admin.get("role", "")):
        raise HTTPException(status_code=403)

    from app.services.document_action_service import DocumentActionService

    _is_sys_admin = is_system_admin(admin.get("role", ""))
    form = await request.form()
    tenant_id_param = form.get("tenant_id", "")

    scoped_tenant = (
        tenant_id_param if _is_sys_admin and tenant_id_param
        else admin.get("tenant_id")
    )

    action_svc = DocumentActionService()
    result = await action_svc.retry_ingest(
        db,
        document_id=document_id,
        tenant_id=scoped_tenant,
        actor_user_id=admin.get("id"),
    )

    tid_qs = f"&tenant_id={tenant_id_param}" if tenant_id_param else ""
    return RedirectResponse(
        f"/dashboard/documents/{document_id}?ok={1 if result.accepted else 0}&msg={result.message}{tid_qs}",
        status_code=303,
    )


@router.post("/dashboard/documents/{document_id}/reindex")
async def dashboard_reindex(
    request: Request,
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_admin(admin.get("role", "")):
        raise HTTPException(status_code=403)

    from app.services.document_action_service import DocumentActionService

    _is_sys_admin = is_system_admin(admin.get("role", ""))
    form = await request.form()
    tenant_id_param = form.get("tenant_id", "")

    scoped_tenant = (
        tenant_id_param if _is_sys_admin and tenant_id_param
        else admin.get("tenant_id")
    )

    action_svc = DocumentActionService()
    result = await action_svc.reindex(
        db,
        document_id=document_id,
        tenant_id=scoped_tenant,
        actor_user_id=admin.get("id"),
    )

    tid_qs = f"&tenant_id={tenant_id_param}" if tenant_id_param else ""
    return RedirectResponse(
        f"/dashboard/documents/{document_id}?ok={1 if result.accepted else 0}&msg={result.message}{tid_qs}",
        status_code=303,
    )


@router.post("/dashboard/documents/{document_id}/resynthesize")
async def dashboard_resynthesize(
    request: Request,
    document_id: int,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_admin(admin.get("role", "")):
        raise HTTPException(status_code=403)

    from app.services.document_action_service import DocumentActionService

    _is_sys_admin = is_system_admin(admin.get("role", ""))
    form = await request.form()
    tenant_id_param = form.get("tenant_id", "")

    scoped_tenant = (
        tenant_id_param if _is_sys_admin and tenant_id_param
        else admin.get("tenant_id")
    )

    action_svc = DocumentActionService()
    result = await action_svc.resynthesize(
        db,
        document_id=document_id,
        tenant_id=scoped_tenant,
        actor_user_id=admin.get("id"),
    )

    tid_qs = f"&tenant_id={tenant_id_param}" if tenant_id_param else ""
    return RedirectResponse(
        f"/dashboard/documents/{document_id}?ok={1 if result.accepted else 0}&msg={result.message}{tid_qs}",
        status_code=303,
    )


# =========================
# OPS DASHBOARD — Phase 8.0
# =========================
@router.get("/dashboard/ops", response_class=HTMLResponse)
async def dashboard_ops(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    if not is_admin(admin.get("role", "")):
        raise HTTPException(status_code=403)

    return templates.TemplateResponse(
        "dashboard/ops.html",
        {
            "request": request,
            "user": admin,
        },
    )



# =========================
# SETTINGS (Plan & Quotas) — Phase 5.0
# =========================
@router.get("/dashboard/settings", response_class=HTMLResponse)
async def dashboard_settings(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    tenant_id = admin.get("tenant_id", "default")

    # Get tenant settings
    setting_repo = TenantSettingRepository()
    setting = await setting_repo.get(db, tenant_id)

    tenant_setting = {
        "plan_code": setting.plan_code if setting else "free",
        "quota_overrides_json": setting.quota_overrides_json if setting else None,
        "enforce_user_rate_limit": setting.enforce_user_rate_limit if setting else False,
    }

    # Resolve effective policy
    policy_svc = QuotaPolicyService()
    policy = await policy_svc.get_effective_policy(db, tenant_id)

    # Get all plans
    plan_repo = PlanRepository()
    plans = await plan_repo.list_all(db)
    plans_list = [{"code": p.code, "name": p.name} for p in plans]

    import json
    overrides_display = ""
    if tenant_setting["quota_overrides_json"]:
        overrides_display = json.dumps(tenant_setting["quota_overrides_json"], indent=2)

    return templates.TemplateResponse(
        "dashboard/settings.html",
        {
            "request": request,
            "user": admin,
            "tenant_id": tenant_id,
            "tenant_setting": tenant_setting,
            "policy": {
                "per_minute": policy.per_minute,
                "burst": policy.burst,
                "token_daily": policy.token_daily,
                "token_monthly": policy.token_monthly,
            },
            "plans": plans_list,
            "quota_overrides_display": overrides_display,
        },
    )


# =========================
# SET LANGUAGE
# =========================
@router.post("/set-language", response_class=HTMLResponse)
async def set_language(
    request: Request,
    language: str = Form("vi"),
):
    """Switch UI language via cookie. Redirects back to referrer."""
    lang = language if language in SUPPORTED_LANGS else DEFAULT_LANG
    referer = request.headers.get("referer", "/dashboard")

    response = RedirectResponse(referer, status_code=303)
    secure = _is_secure_cookie()
    response.set_cookie(
        "lang",
        lang,
        path="/",
        max_age=365 * 24 * 3600,  # 1 year
        httponly=False,  # JS-readable for client-side i18n if needed
        secure=secure,
        samesite="lax",
    )
    return response


# =========================
# LOGOUT
# =========================
@router.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=302)
    secure = _is_secure_cookie()
    # Security: delete with the SAME path used during set_cookie.
    # No domain= (matches exact origin). httponly/secure match set_cookie policy.
    response.delete_cookie(
        "access_token",
        path="/",
        httponly=True,
        secure=secure,
        samesite="lax",
    )
    # Also clear CSRF cookie on logout
    response.delete_cookie(
        "csrf_token",
        path="/",
        secure=secure,
        samesite="lax",
    )
    return response
