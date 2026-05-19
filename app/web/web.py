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
from app.services.api_key_service import APIKeyService
from app.services.user_service import UserService
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

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "query_kpis": None,
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

    api_keys = await APIKeyService.list_by_user_id(db, user_id)

    return templates.TemplateResponse(
        "dashboard/user_detail.html",
        {
            "request": request,
            "user": admin,
            "target_user": detail["user"],
            "quota": None,
            "api_keys": api_keys,
            "recent_usage": detail["recent_usage"][:20],
            "user_query_stats": None,
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

    keys = []
    target_user = None
    filter_error = None

    if user_id:
        actor = await UserService.get_by_id(db, admin["id"])
        detail = await AdminUserService.get_user_detail(db, user_id, actor=actor)
        if not detail:
            filter_error = "User not found or not accessible."
        else:
            target_user = detail["user"]
            keys = await APIKeyService.list_by_user_id(db, user_id)

    return templates.TemplateResponse(
        "dashboard/api_keys.html",
        {
            "request": request,
            "user": admin,
            "api_keys": keys,
            "filter_user_id": user_id,
            "target_user": target_user,
            "filter_error": filter_error,
        },
    )


# =========================
# RAG CENTER - Phase R6.4-UI-1
# =========================
@router.get("/dashboard/rag", response_class=HTMLResponse)
async def dashboard_rag_center(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    metrics = [
        {"label": "Tổng API Keys", "value": "24", "variant": "card-accent"},
        {"label": "API Keys đang hoạt động", "value": "18", "variant": "card-success"},
        {"label": "Requests hôm nay", "value": "1,284", "variant": ""},
        {"label": "Lỗi hôm nay", "value": "7", "variant": "card-danger"},
        {"label": "Mapping drafts", "value": "12", "variant": "card-warning"},
        {"label": "Documents indexed", "value": "3,420", "variant": ""},
    ]

    system_statuses = [
        {"label": "RAG API", "status": "Online", "badge": "badge-success"},
        {"label": "CTĐT API", "status": "Available", "badge": "badge-success"},
        {"label": "Vector Index", "status": "Ready", "badge": "badge-primary"},
        {"label": "Embedding Provider", "status": "Configured", "badge": "badge-accent"},
        {"label": "Mapping Draft Engine", "status": "Available", "badge": "badge-success"},
    ]

    quick_actions = [
        {
            "label": "Quản lý API Keys",
            "href": "/dashboard/api-keys",
            "description": "Tạo, xoay vòng và thu hồi API key quản trị.",
            "disabled": False,
        },
        {
            "label": "Xem Usage Logs",
            "href": "/dashboard/rag/usage",
            "description": "Theo dõi request, lỗi và hoạt động truy vấn.",
            "disabled": False,
        },
        {
            "label": "CTĐT API Catalog",
            "href": "/dashboard/rag/ctdt",
            "description": "Tổng hợp endpoint CTĐT quan trọng.",
            "disabled": False,
        },
        {
            "label": "Cấu hình RAG",
            "href": "/dashboard/rag/settings",
            "description": "Thiết lập provider, index và pipeline RAG.",
            "disabled": False,
        },
    ]

    recent_activities = [
        {
            "time": "09:42",
            "actor": "system_admin",
            "action": "Build mapping draft",
            "target": "update_cycle: 2026-CTDT-01",
            "status": "Success",
            "badge": "badge-success",
        },
        {
            "time": "09:17",
            "actor": "tenant_admin",
            "action": "Analyze cycle",
            "target": "Khoa CNTT",
            "status": "Queued",
            "badge": "badge-warning",
        },
        {
            "time": "08:55",
            "actor": "api_key: rag-demo",
            "action": "Query",
            "target": "/api/v1/query",
            "status": "Success",
            "badge": "badge-success",
        },
        {
            "time": "08:21",
            "actor": "system",
            "action": "Index document",
            "target": "CTĐT handbook",
            "status": "Ready",
            "badge": "badge-primary",
        },
        {
            "time": "07:48",
            "actor": "api_key: assistant-demo",
            "action": "Assistant request",
            "target": "/api/v1/assistant/respond",
            "status": "Error",
            "badge": "badge-danger",
        },
    ]

    ctdt_endpoints = [
        {
            "method": "GET",
            "path": "/api/v1/ctdt/update-cycles/{update_cycle_id}/mapping-draft/latest",
        },
        {"method": "POST", "path": "/api/v1/ctdt/update-cycles/mapping-draft/build"},
        {"method": "POST", "path": "/api/v1/ctdt/update-cycles/analyze"},
        {"method": "POST", "path": "/api/v1/query"},
        {"method": "POST", "path": "/api/v1/assistant/respond"},
    ]

    return templates.TemplateResponse(
        "dashboard/rag.html",
        {
            "request": request,
            "user": admin,
            "metrics": metrics,
            "system_statuses": system_statuses,
            "quick_actions": quick_actions,
            "recent_activities": recent_activities,
            "ctdt_endpoints": ctdt_endpoints,
        },
    )


# =========================
# CTDT API CATALOG - Phase R6.4-UI-3
# =========================
@router.get("/dashboard/rag/ctdt", response_class=HTMLResponse)
async def dashboard_rag_ctdt_catalog(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    catalog = [
        {
            "method": "GET",
            "path": "/api/v1/ctdt/update-cycles/{update_cycle_id}/mapping-draft/latest",
            "group": "Mapping Draft",
            "purpose": "Lấy bản mapping draft mới nhất theo update cycle",
            "status": "Available",
            "auth": "API Key/JWT",
            "notes": "R6.3C",
        },
        {
            "method": "POST",
            "path": "/api/v1/ctdt/update-cycles/mapping-draft/build",
            "group": "Mapping Draft",
            "purpose": "Build mapping draft từ dữ liệu phân tích",
            "status": "Available",
            "auth": "API Key/JWT",
            "notes": "R6.3B — Dùng cho luồng tạo draft",
        },
        {
            "method": "POST",
            "path": "/api/v1/ctdt/update-cycles/analyze",
            "group": "CTĐT Analysis",
            "purpose": "Phân tích một đợt cập nhật CTĐT",
            "status": "Available",
            "auth": "API Key/JWT",
            "notes": "Đầu vào cho review và mapping",
        },
        {
            "method": "POST",
            "path": "/api/v1/query",
            "group": "Query",
            "purpose": "Truy vấn RAG trực tiếp",
            "status": "Available",
            "auth": "API Key/JWT",
            "notes": "Endpoint RAG chính",
        },
        {
            "method": "POST",
            "path": "/api/v1/assistant/respond",
            "group": "Assistant",
            "purpose": "Chat assistant dùng RAG context",
            "status": "Available",
            "auth": "API Key/JWT",
            "notes": "Dùng cho trải nghiệm hội thoại",
        },
        {
            "method": "POST",
            "path": "/api/v1/documents/upsert",
            "group": "Admin",
            "purpose": "Upsert tài liệu vào corpus RAG",
            "status": "Available",
            "auth": "JWT",
            "notes": "Idempotent document upsert",
        },
        {
            "method": "POST",
            "path": "/api/v1/documents/upload",
            "group": "Admin",
            "purpose": "Upload/ingest tài liệu",
            "status": "Available",
            "auth": "JWT",
            "notes": "Nguồn dữ liệu index",
        },
        {
            "method": "GET",
            "path": "/api/v1/admin/users",
            "group": "Admin",
            "purpose": "Quản lý user",
            "status": "Available",
            "auth": "JWT",
            "notes": "Admin dashboard/API",
        },
        {
            "method": "POST",
            "path": "/api/v1/auth/login",
            "group": "Auth",
            "purpose": "Đăng nhập lấy token",
            "status": "Available",
            "auth": "Public",
            "notes": "Web login dùng cookie/JWT",
        },
        {
            "method": "GET",
            "path": "/health/live",
            "group": "Health",
            "purpose": "Health check service",
            "status": "Available",
            "auth": "None",
            "notes": "Dùng cho uptime probe",
        },
        {
            "method": "GET",
            "path": "/api/v1/ctdt/openapi/catalog",
            "group": "CTĐT Analysis",
            "purpose": "Catalog endpoint tự động từ OpenAPI",
            "status": "Planned",
            "auth": "API Key/JWT",
            "notes": "Phase sau",
        },
        {
            "method": "POST",
            "path": "/api/v1/ctdt/mapping-draft/export",
            "group": "Mapping Draft",
            "purpose": "Export mapping draft cho Laravel CTĐT",
            "status": "Planned",
            "auth": "API Key/JWT",
            "notes": "Chưa triển khai",
        },
    ]

    metrics = [
        {"label": "Total endpoints", "value": str(len(catalog)), "variant": "card-accent"},
        {
            "label": "CTĐT endpoints",
            "value": str(sum(1 for item in catalog if item["group"] == "CTĐT Analysis")),
            "variant": "card-success",
        },
        {
            "label": "Mapping Draft endpoints",
            "value": str(sum(1 for item in catalog if item["group"] == "Mapping Draft")),
            "variant": "card-warning",
        },
        {
            "label": "Query/Assistant endpoints",
            "value": str(sum(1 for item in catalog if item["group"] in ("Query", "Assistant"))),
            "variant": "",
        },
        {
            "label": "Available endpoints",
            "value": str(sum(1 for item in catalog if item["status"] == "Available")),
            "variant": "card-success",
        },
    ]

    group_options = [
        "All groups",
        "CTĐT Analysis",
        "Mapping Draft",
        "Query",
        "Assistant",
        "Admin",
    ]
    method_options = ["All methods", "GET", "POST"]
    status_options = ["All", "Available", "Planned", "Deprecated"]
    integration_notes = [
        "Laravel CTĐT có thể gọi các endpoint này qua RAG_BASE_URL.",
        "API key thật sẽ được quản lý qua API Keys page.",
        "Mapping draft latest endpoint đã có từ R6.3C.",
        "Phase này chỉ là catalog UI, chưa tự introspect OpenAPI.",
    ]

    return templates.TemplateResponse(
        "dashboard/rag_ctdt.html",
        {
            "request": request,
            "user": admin,
            "metrics": metrics,
            "catalog": catalog,
            "group_options": group_options,
            "method_options": method_options,
            "status_options": status_options,
            "integration_notes": integration_notes,
        },
    )


# =========================
# RAG SETTINGS / HEALTH - Phase R6.4-UI-4
# =========================
@router.get("/dashboard/rag/settings", response_class=HTMLResponse)
async def dashboard_rag_settings(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    from urllib.parse import urlsplit, urlunsplit
    from app.core.config import settings as app_settings

    def raw_config(name: str):
        return getattr(app_settings, name, None)

    def display_config(name: str, default: str = "N/A") -> str:
        value = raw_config(name)
        if value is None:
            return default
        if isinstance(value, str):
            value = value.strip()
            return value if value else default
        return str(value)

    def secret_status(name: str) -> dict[str, str]:
        value = raw_config(name)
        configured = bool(value.strip()) if isinstance(value, str) else value is not None
        return {
            "value": "Configured" if configured else "Not configured",
            "badge": "badge-success" if configured else "badge-danger",
        }

    def status_badge(status: str) -> str:
        return {
            "Ready": "badge-success",
            "Configured": "badge-success",
            "Not configured": "badge-danger",
            "Planned": "badge-warning",
            "Unknown": "badge-muted",
        }.get(status, "badge-muted")

    def safe_url(value: str) -> str:
        if not value or value in ("N/A", "Not configured"):
            return "Not configured"
        try:
            parsed = urlsplit(value)
            if parsed.scheme and parsed.netloc:
                host = parsed.hostname or ""
                port = f":{parsed.port}" if parsed.port else ""
                return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))
        except Exception:
            return "Configured"
        return value

    app_name = display_config("APP_NAME")
    env = display_config("ENV")
    embedding_provider = display_config("EMBEDDING_PROVIDER", "Not configured")
    embedding_model = display_config("EMBEDDING_MODEL", "Not configured")
    embedding_dim = display_config("EMBEDDING_DIM", "N/A")
    vector_index = display_config("VECTOR_INDEX", "Not configured")
    retrieval_mode = (
        display_config("RETRIEVAL_MODE", "")
        or display_config("RETRIEVAL_REPRESENTATION_MODE", "N/A")
    )
    query_max_k = display_config("QUERY_MAX_K", "") or display_config("QUERY_FINAL_LIMIT", "N/A")
    rag_base_url = display_config("RAG_BASE_URL", "") or display_config("PUBLIC_BASE_URL", "N/A")
    qdrant_url = safe_url(display_config("QDRANT_URL", ""))
    openai_secret = secret_status("OPENAI_API_KEY")
    database_secret = secret_status("DATABASE_URL")
    jwt_secret_name = "JWT_SECRET" if raw_config("JWT_SECRET") is not None else "SECRET_KEY"
    jwt_secret = secret_status(jwt_secret_name)

    vector_status = (
        "Not configured"
        if vector_index.lower() in ("", "n/a", "not configured", "none", "null")
        else "Ready"
    )
    embedding_status = (
        "Not configured"
        if embedding_provider.lower() in ("", "n/a", "not configured", "none")
        else "Configured"
    )

    metrics = [
        {"label": "Environment", "value": env, "variant": "card-accent"},
        {"label": "Embedding Provider", "value": embedding_provider, "variant": "card-success" if embedding_status == "Configured" else "card-danger"},
        {"label": "Vector Index", "value": vector_index, "variant": "card-success" if vector_status == "Ready" else "card-danger"},
        {"label": "Retrieval Mode", "value": retrieval_mode, "variant": ""},
        {"label": "API Key Secret", "value": openai_secret["value"], "variant": "card-success" if openai_secret["value"] == "Configured" else "card-danger"},
        {"label": "Database", "value": database_secret["value"], "variant": "card-success" if database_secret["value"] == "Configured" else "card-danger"},
    ]

    runtime_config = [
        {"key": "APP_NAME", "value": app_name, "notes": "Tên runtime service."},
        {"key": "ENV", "value": env, "notes": "Môi trường triển khai hiện tại."},
        {"key": "EMBEDDING_PROVIDER", "value": embedding_provider, "notes": "Provider sinh embedding."},
        {"key": "EMBEDDING_MODEL", "value": embedding_model, "notes": "Model embedding đang cấu hình."},
        {"key": "EMBEDDING_DIM", "value": embedding_dim, "notes": "Số chiều vector embedding."},
        {"key": "VECTOR_INDEX", "value": vector_index, "notes": "Backend vector index."},
        {"key": "RETRIEVAL_MODE", "value": retrieval_mode, "notes": "Chế độ retrieval mặc định."},
        {"key": "QUERY_MAX_K", "value": query_max_k, "notes": "Fallback từ QUERY_FINAL_LIMIT nếu QUERY_MAX_K chưa có."},
        {"key": "RAG_BASE_URL / PUBLIC_BASE_URL", "value": rag_base_url, "notes": "URL Laravel CTĐT gọi đến RAG service."},
        {"key": "OPENAI_API_KEY", "value": openai_secret["value"], "notes": "Secret được mask, không render key thật."},
        {"key": "QDRANT_URL", "value": qdrant_url, "notes": "URL hiển thị đã loại bỏ credential nếu có."},
        {"key": "DATABASE_URL", "value": database_secret["value"], "notes": "Connection string được mask."},
    ]

    provider_health = [
        {"name": "Embedding provider", "status": embedding_status, "badge": status_badge(embedding_status), "details": embedding_provider},
        {"name": "Vector index", "status": vector_status, "badge": status_badge(vector_status), "details": vector_index},
        {"name": "Database", "status": database_secret["value"], "badge": database_secret["badge"], "details": "Connection string masked"},
        {"name": "CTĐT API router", "status": "Ready", "badge": status_badge("Ready"), "details": "/api/v1/ctdt"},
        {"name": "Mapping Draft Engine", "status": "Ready", "badge": status_badge("Ready"), "details": "R6.3C latest draft endpoint available"},
        {"name": "Usage logging", "status": "Planned", "badge": status_badge("Planned"), "details": "Mock UI only in this phase"},
    ]

    safe_secrets = [
        {"key": "OPENAI_API_KEY", "status": openai_secret["value"], "badge": openai_secret["badge"]},
        {"key": "DATABASE_URL", "status": database_secret["value"], "badge": database_secret["badge"]},
        {"key": "JWT_SECRET / SECRET_KEY", "status": jwt_secret["value"], "badge": jwt_secret["badge"]},
    ]

    integration_notes = [
        "Laravel CTĐT nên gọi RAG qua RAG_BASE_URL.",
        "API Keys page quản lý key truy cập.",
        "Usage Logs page sẽ nối dữ liệu thật ở phase sau.",
        "CTĐT APIs page hiện là static catalog.",
        "Thay đổi config thật nên qua .env/deployment, không sửa trực tiếp trên UI ở phase này.",
    ]

    readonly_settings = [
        {"label": "RAG endpoint URL", "value": rag_base_url},
        {"label": "Default retrieval mode", "value": retrieval_mode},
        {"label": "Max query K", "value": query_max_k},
        {"label": "Enable usage logging", "value": "Planned"},
        {"label": "Enable mapping draft access", "value": "Enabled"},
        {"label": "Embedding provider", "value": embedding_provider},
        {"label": "Vector index", "value": vector_index},
    ]

    return templates.TemplateResponse(
        "dashboard/rag_settings.html",
        {
            "request": request,
            "user": admin,
            "metrics": metrics,
            "runtime_config": runtime_config,
            "provider_health": provider_health,
            "safe_secrets": safe_secrets,
            "integration_notes": integration_notes,
            "readonly_settings": readonly_settings,
        },
    )


# =========================
# RAG USAGE LOGS - Phase R6.4-UI-2
# =========================
@router.get("/dashboard/rag/usage", response_class=HTMLResponse)
async def dashboard_rag_usage(
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        admin = await get_admin_user_from_cookie(request, db)
    except Exception:
        return RedirectResponse("/login")

    metrics = [
        {"label": "Requests hôm nay", "value": "1,284", "variant": "card-accent"},
        {"label": "Success rate", "value": "98.6%", "variant": "card-success"},
        {"label": "Avg latency", "value": "184 ms", "variant": ""},
        {"label": "Token usage hôm nay", "value": "248K", "variant": "card-warning"},
        {"label": "Errors hôm nay", "value": "7", "variant": "card-danger"},
    ]

    endpoint_options = [
        "All endpoints",
        "/api/v1/query",
        "/api/v1/assistant/respond",
        "/api/v1/ctdt/update-cycles/analyze",
        "/api/v1/ctdt/update-cycles/mapping-draft/build",
        "/api/v1/ctdt/update-cycles/{id}/mapping-draft/latest",
    ]

    status_options = ["All", "Success", "Error"]

    usage_logs = [
        {
            "time": "2026-05-18 09:58:12",
            "actor": "api_key: rag-demo",
            "method": "POST",
            "endpoint": "/api/v1/query",
            "status_code": 200,
            "result": "Success",
            "badge": "badge-success",
            "latency": "142 ms",
            "tokens": "1,248",
            "ip": "10.12.0.24",
            "error_code": "-",
        },
        {
            "time": "2026-05-18 09:55:43",
            "actor": "api_key: assistant-demo",
            "method": "POST",
            "endpoint": "/api/v1/assistant/respond",
            "status_code": 200,
            "result": "Success",
            "badge": "badge-success",
            "latency": "231 ms",
            "tokens": "2,904",
            "ip": "10.12.0.31",
            "error_code": "-",
        },
        {
            "time": "2026-05-18 09:51:08",
            "actor": "tenant_admin",
            "method": "POST",
            "endpoint": "/api/v1/ctdt/update-cycles/analyze",
            "status_code": 200,
            "result": "Success",
            "badge": "badge-success",
            "latency": "618 ms",
            "tokens": "8,420",
            "ip": "10.12.1.10",
            "error_code": "-",
        },
        {
            "time": "2026-05-18 09:44:37",
            "actor": "system_admin",
            "method": "POST",
            "endpoint": "/api/v1/ctdt/update-cycles/mapping-draft/build",
            "status_code": 200,
            "result": "Success",
            "badge": "badge-success",
            "latency": "734 ms",
            "tokens": "12,118",
            "ip": "10.12.1.12",
            "error_code": "-",
        },
        {
            "time": "2026-05-18 09:39:20",
            "actor": "tenant_admin",
            "method": "GET",
            "endpoint": "/api/v1/ctdt/update-cycles/{id}/mapping-draft/latest",
            "status_code": 200,
            "result": "Success",
            "badge": "badge-success",
            "latency": "86 ms",
            "tokens": "0",
            "ip": "10.12.1.10",
            "error_code": "-",
        },
        {
            "time": "2026-05-18 09:32:02",
            "actor": "api_key: rag-demo",
            "method": "POST",
            "endpoint": "/api/v1/query",
            "status_code": 422,
            "result": "Error",
            "badge": "badge-danger",
            "latency": "39 ms",
            "tokens": "0",
            "ip": "10.12.0.24",
            "error_code": "VALIDATION_ERROR",
        },
        {
            "time": "2026-05-18 09:18:49",
            "actor": "anonymous",
            "method": "POST",
            "endpoint": "/api/v1/assistant/respond",
            "status_code": 401,
            "result": "Error",
            "badge": "badge-danger",
            "latency": "21 ms",
            "tokens": "0",
            "ip": "203.0.113.44",
            "error_code": "UNAUTHORIZED",
        },
        {
            "time": "2026-05-18 09:07:16",
            "actor": "api_key: assistant-demo",
            "method": "POST",
            "endpoint": "/api/v1/assistant/respond",
            "status_code": 500,
            "result": "Error",
            "badge": "badge-danger",
            "latency": "1,840 ms",
            "tokens": "512",
            "ip": "10.12.0.31",
            "error_code": "PROVIDER_TIMEOUT",
        },
        {
            "time": "2026-05-18 08:56:54",
            "actor": "api_key: ctdt-batch",
            "method": "POST",
            "endpoint": "/api/v1/ctdt/update-cycles/mapping-draft/build",
            "status_code": 200,
            "result": "Success",
            "badge": "badge-success",
            "latency": "691 ms",
            "tokens": "10,772",
            "ip": "10.12.2.5",
            "error_code": "-",
        },
        {
            "time": "2026-05-18 08:43:11",
            "actor": "tenant_admin",
            "method": "POST",
            "endpoint": "/api/v1/ctdt/update-cycles/analyze",
            "status_code": 422,
            "result": "Error",
            "badge": "badge-danger",
            "latency": "47 ms",
            "tokens": "0",
            "ip": "10.12.1.10",
            "error_code": "MISSING_UPDATE_CYCLE",
        },
        {
            "time": "2026-05-18 08:31:26",
            "actor": "api_key: rag-demo",
            "method": "POST",
            "endpoint": "/api/v1/query",
            "status_code": 200,
            "result": "Success",
            "badge": "badge-success",
            "latency": "156 ms",
            "tokens": "1,034",
            "ip": "10.12.0.24",
            "error_code": "-",
        },
    ]

    return templates.TemplateResponse(
        "dashboard/rag_usage.html",
        {
            "request": request,
            "user": admin,
            "metrics": metrics,
            "endpoint_options": endpoint_options,
            "status_options": status_options,
            "usage_logs": usage_logs,
        },
    )


# =========================
# USAGE
# =========================
@router.get("/dashboard/usage", response_class=HTMLResponse)
async def dashboard_usage(request: Request, db: AsyncSession = Depends(get_db)):
    return RedirectResponse("/dashboard", status_code=302)


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
async def dashboard_tenants(request: Request, db: AsyncSession = Depends(get_db)):
    return RedirectResponse("/dashboard", status_code=302)


# =========================
# NEW TENANT (FORM)
# =========================
@router.get("/dashboard/tenants/new", response_class=HTMLResponse)
async def dashboard_tenant_new(request: Request, db: AsyncSession = Depends(get_db)):
    return RedirectResponse("/dashboard", status_code=302)


# =========================
# CREATE TENANT (POST) — CSRF-protected
# =========================
@router.post("/dashboard/tenants/new", response_class=HTMLResponse)
async def dashboard_tenant_create(request: Request, db: AsyncSession = Depends(get_db)):
    return RedirectResponse("/dashboard", status_code=302)


# =========================
# TENANT DETAIL
# =========================
@router.get("/dashboard/tenants/{tenant_id}", response_class=HTMLResponse)
async def dashboard_tenant_detail(tenant_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    return RedirectResponse("/dashboard", status_code=302)


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
async def dashboard_ops(request: Request, db: AsyncSession = Depends(get_db)):
    return RedirectResponse("/dashboard", status_code=302)



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

    return templates.TemplateResponse(
        "dashboard/settings.html",
        {
            "request": request,
            "user": admin,
            "tenant_id": tenant_id,
            "tenant_setting": {"plan_code": "ctdt", "quota_overrides_json": None, "enforce_user_rate_limit": False},
            "policy": {"per_minute": 0, "burst": 0, "token_daily": 0, "token_monthly": 0},
            "plans": [],
            "quota_overrides_display": "",
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
