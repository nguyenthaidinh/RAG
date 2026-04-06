import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import settings
from app.core.usage_middleware import UsageLoggingMiddleware
from app.core.request_id_middleware import RequestIdMiddleware
from app.core.metrics import PrometheusMiddleware

from app.api.v1.auth import router as auth_router
from app.api.v1.system import router as system_router
from app.api.v1.documents import router as documents_router
from app.api.v1.admin_users import router as admin_users_router
from app.api.v1.admin_quotas import router as admin_quotas_router
from app.api.v1.admin_api_keys import router as admin_api_keys_router
from app.api.v1.admin_usage import router as admin_usage_router
from app.api.v1.admin_tenants import router as admin_tenants_router
from app.api.v1.admin_query_analytics import router as admin_query_analytics_router
from app.api.v1.admin_query_export import router as admin_query_export_router
from app.api.v1.admin_quota_policy import router as admin_quota_policy_router
from app.api.v1.admin_audit_events import router as admin_audit_events_router
from app.api.v1.admin_documents import router as admin_documents_router
from app.api.v1.admin_source_platform import router as admin_source_platform_router
from app.api.v1.query import router as query_router
from app.api.v1.assistant import router as assistant_router
from app.api.v1.metrics import router as metrics_router
from app.api.v1.health import router as health_router
from app.api.v1.ops import router as ops_router
from app.api.v1.system_context_debug import router as system_context_debug_router
from app.web.web import router as web_router

logger = logging.getLogger(__name__)


# ── Phase 8.0: Lifespan (graceful shutdown) ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan: startup + shutdown hooks.

    Startup:  initialize shutdown manager (resets uptime clock).
    Shutdown: drain in-flight requests, close DB pool.
    """
    # ── Startup ───────────────────────────────────────────────────
    if settings.GRACEFUL_SHUTDOWN_ENABLED:
        from app.core.shutdown import get_shutdown_manager
        get_shutdown_manager()  # initialize / reset uptime clock
        logger.info("phase8.startup graceful_shutdown=enabled")

    if settings.BACKPRESSURE_ENABLED:
        from app.core.backpressure import get_backpressure_manager
        get_backpressure_manager()  # initialize
        logger.info(
            "phase8.startup backpressure=enabled global_max=%d per_tenant_max=%d",
            settings.BACKPRESSURE_MAX_CONCURRENT_GLOBAL,
            settings.BACKPRESSURE_MAX_CONCURRENT_PER_TENANT,
        )

    # Start usage log dispatcher (bounded queue worker)
    from app.core.usage_dispatcher import get_usage_dispatcher
    _usage_disp = get_usage_dispatcher()
    await _usage_disp.start()

    # Start source sync scheduler (Phase 7) if enabled
    _sync_scheduler = None
    if settings.SOURCE_SYNC_SCHEDULER_ENABLED:
        from app.services.source_platform.source_sync_scheduler import (
            get_sync_scheduler,
        )
        _sync_scheduler = get_sync_scheduler()
        _sync_scheduler.start()
        logger.info("phase7.startup source_sync_scheduler=enabled")

    yield

    # ── Shutdown ──────────────────────────────────────────────────
    # Stop source sync scheduler first (Phase 7)
    if _sync_scheduler is not None:
        await _sync_scheduler.stop()

    # Stop usage dispatcher (drain remaining items)
    await _usage_disp.stop()

    if settings.GRACEFUL_SHUTDOWN_ENABLED:
        from app.core.shutdown import shutdown_sequence
        await shutdown_sequence()


# ── Phase 8.0: Shutdown-aware middleware ──────────────────────────────

class ShutdownAwareMiddleware(BaseHTTPMiddleware):
    """
    Track in-flight requests and reject new ones during drain.

    - Registers each request with the ShutdownManager.
    - During drain: returns 503 to new requests immediately.
    - Health endpoints are always allowed (for K8s probes).
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.GRACEFUL_SHUTDOWN_ENABLED:
            return await call_next(request)

        # Always allow health checks (K8s probes must work during drain)
        path = request.url.path
        if path.startswith("/health/"):
            return await call_next(request)

        from app.core.shutdown import get_shutdown_manager
        mgr = get_shutdown_manager()

        if not mgr.request_started():
            return JSONResponse(
                {
                    "error": "server_shutting_down",
                    "message": "Server is shutting down. Please retry.",
                },
                status_code=503,
            )

        try:
            response = await call_next(request)
            return response
        finally:
            mgr.request_finished()


# ── App creation ──────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    redirect_slashes=False,
)

# Static files
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Middleware (order matters — outermost first, registered last runs first)
# RequestIdMiddleware must be registered LAST so it executes FIRST,
# ensuring all downstream middleware/routes see a consistent request_id.
app.add_middleware(UsageLoggingMiddleware)
app.add_middleware(PrometheusMiddleware)
app.add_middleware(ShutdownAwareMiddleware)  # Phase 8.0
app.add_middleware(RequestIdMiddleware)

# Health checks (Phase 8.0 — no auth required, K8s compatible)
app.include_router(health_router)

# API
app.include_router(auth_router)
app.include_router(system_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(assistant_router)

# Admin API
app.include_router(admin_users_router)
app.include_router(admin_quotas_router)
app.include_router(admin_api_keys_router)
app.include_router(admin_usage_router)
app.include_router(admin_tenants_router)
app.include_router(admin_query_analytics_router)
app.include_router(admin_query_export_router)
app.include_router(admin_quota_policy_router)
app.include_router(admin_audit_events_router)
app.include_router(admin_documents_router)
app.include_router(admin_source_platform_router)

# System Context (Phase 1 — debug only)
app.include_router(system_context_debug_router)

# Observability (Phase 7.0)
app.include_router(metrics_router)

# Ops (Phase 8.0)
app.include_router(ops_router)

# Web (HTML pages)
app.include_router(web_router)