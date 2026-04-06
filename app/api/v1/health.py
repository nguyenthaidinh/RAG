"""
Health check endpoints (Phase 8.0).

/health/live  — livenessProbe  (K8s compatible)
/health/ready — readinessProbe (K8s compatible)

Rules:
  - /health/live returns 200 if the process is running; checks NOTHING else.
  - /health/ready checks DB + vector index with strict timeouts.
  - If ANY dependency is unavailable → 503.
  - Health endpoints MUST NEVER block request threads.
  - Health endpoints MUST NOT perform heavy queries.
  - Gated behind HEALTHCHECKS_ENABLED flag.
"""
from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter
from starlette.responses import JSONResponse

from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


# ── /health/live ──────────────────────────────────────────────────────

@router.get("/health/live")
async def liveness() -> JSONResponse:
    """
    Liveness probe: process is running.

    Does NOT check DB or vector index.
    Used for crash detection / restart.
    """
    if not settings.HEALTHCHECKS_ENABLED:
        return JSONResponse({"status": "disabled"}, status_code=200)

    return JSONResponse({"status": "alive"}, status_code=200)


# ── /health/ready ─────────────────────────────────────────────────────

async def _check_db(timeout_s: float) -> str:
    """Check PostgreSQL connectivity with strict timeout."""
    try:
        from app.db.session import AsyncSessionLocal
        async with AsyncSessionLocal() as session:
            await asyncio.wait_for(
                session.execute(
                    __import__("sqlalchemy").text("SELECT 1")
                ),
                timeout=timeout_s,
            )
        return "ok"
    except asyncio.TimeoutError:
        logger.warning("health.db_check timed out after %.0fms", timeout_s * 1000)
        return "timeout"
    except Exception as exc:
        logger.warning("health.db_check failed: %s", exc)
        return "down"


async def _check_vector(timeout_s: float) -> str:
    """Check vector index availability with strict timeout."""
    try:
        idx_type = settings.VECTOR_INDEX.lower().strip()

        if idx_type == "null":
            # NullIndex is always available
            return "ok"

        if idx_type == "qdrant":
            try:
                from qdrant_client import AsyncQdrantClient
            except ImportError:
                return "ok"  # not installed = not used

            client = AsyncQdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY or None,
            )
            try:
                await asyncio.wait_for(
                    client.get_collections(),
                    timeout=timeout_s,
                )
                return "ok"
            finally:
                await client.close()

        if idx_type == "pgvector":
            # pgvector uses the same DB — already checked
            return "ok"

        if idx_type == "faiss":
            # In-process, always available if loaded
            return "ok"

        return "ok"

    except asyncio.TimeoutError:
        logger.warning("health.vector_check timed out after %.0fms", timeout_s * 1000)
        return "timeout"
    except Exception as exc:
        logger.warning("health.vector_check failed: %s", exc)
        return "down"


@router.get("/health/ready")
async def readiness() -> JSONResponse:
    """
    Readiness probe: all dependencies reachable.

    Checks PostgreSQL and vector index with strict timeouts.
    Returns 503 if ANY dependency is unavailable.
    """
    if not settings.HEALTHCHECKS_ENABLED:
        return JSONResponse({"status": "disabled"}, status_code=200)

    db_timeout_s = settings.HEALTH_DB_TIMEOUT_MS / 1000.0
    vec_timeout_s = settings.HEALTH_VECTOR_TIMEOUT_MS / 1000.0

    # Run checks concurrently
    db_status, vec_status = await asyncio.gather(
        _check_db(db_timeout_s),
        _check_vector(vec_timeout_s),
    )

    all_ok = db_status == "ok" and vec_status == "ok"

    status = "ready" if all_ok else "degraded"
    http_status = 200 if all_ok else 503

    return JSONResponse(
        {
            "status": status,
            "db": db_status,
            "vector_index": vec_status,
        },
        status_code=http_status,
    )
