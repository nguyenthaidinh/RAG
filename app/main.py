import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.request_id_middleware import RequestIdMiddleware
from app.core.metrics import PrometheusMiddleware

from app.api.v1.auth import router as auth_router
from app.api.v1.system import router as system_router
from app.api.v1.documents import router as documents_router
from app.api.v1.admin_users import router as admin_users_router
from app.api.v1.query import router as query_router
from app.api.v1.assistant import router as assistant_router
from app.api.v1.health import router as health_router
from app.api.v1.ctdt import router as ctdt_router
from app.web.web import router as web_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: startup + shutdown hooks."""
    logger.info("ctdt_ai_server.startup")
    yield
    logger.info("ctdt_ai_server.shutdown")


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    redirect_slashes=False,
)

# Static files
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Middleware
app.add_middleware(PrometheusMiddleware)
app.add_middleware(RequestIdMiddleware)

# Health checks (no auth required)
app.include_router(health_router)

# Core API
app.include_router(auth_router)
app.include_router(system_router)
app.include_router(documents_router)
app.include_router(query_router)
app.include_router(assistant_router)

# Admin API (simplified)
app.include_router(admin_users_router)

# CTDT-specific endpoint
app.include_router(ctdt_router)

# Web UI (HTML pages for testing)
app.include_router(web_router)
