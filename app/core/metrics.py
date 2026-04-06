"""
Prometheus metrics registry (Phase 7.0).

All metrics are exposed through a dedicated CollectorRegistry so that
the default process/platform collectors do not pollute the /metrics
output with unexpected data.

Rules:
  - Non-blocking, best-effort
  - No raw text in labels
  - High-cardinality labels FORBIDDEN (no user_id, request_id, etc.)
  - All metrics are gated behind ``METRICS_ENABLED``
"""
from __future__ import annotations

import logging
import re
import time
from typing import Callable

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Registry ──────────────────────────────────────────────────────────

REGISTRY = CollectorRegistry()

# ── A. HTTP / API Metrics ─────────────────────────────────────────────

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    registry=REGISTRY,
)

# ── B. AI / Query Metrics ────────────────────────────────────────────

AI_QUERIES_TOTAL = Counter(
    "ai_queries_total",
    "Total AI queries",
    ["tenant_id", "mode", "status"],
    registry=REGISTRY,
)

AI_QUERY_LATENCY = Histogram(
    "ai_query_latency_seconds",
    "AI query latency in seconds",
    ["tenant_id", "mode"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

AI_TOKENS_USED_TOTAL = Counter(
    "ai_tokens_used_total",
    "Total tokens consumed",
    ["tenant_id", "type"],
    registry=REGISTRY,
)

# ── C. Quota & Rate-limit Metrics ────────────────────────────────────

RATE_LIMIT_HITS_TOTAL = Counter(
    "rate_limit_hits_total",
    "Total rate limit hits",
    ["tenant_id", "scope"],
    registry=REGISTRY,
)

TOKEN_QUOTA_EXCEEDED_TOTAL = Counter(
    "token_quota_exceeded_total",
    "Total token quota exceeded events",
    ["tenant_id"],
    registry=REGISTRY,
)

# ── D. Infrastructure Metrics ────────────────────────────────────────

DB_POOL_SIZE = Gauge(
    "db_pool_size",
    "Database connection pool size",
    registry=REGISTRY,
)

DB_POOL_CHECKED_IN = Gauge(
    "db_pool_checked_in",
    "Database connections checked in (available)",
    registry=REGISTRY,
)

DB_POOL_CHECKED_OUT = Gauge(
    "db_pool_checked_out",
    "Database connections checked out (in use)",
    registry=REGISTRY,
)

VECTOR_QUERY_LATENCY = Histogram(
    "vector_index_query_latency_seconds",
    "Vector index query latency in seconds",
    registry=REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
)


# ── Path normalizer ──────────────────────────────────────────────────

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_INT_ID_RE = re.compile(r"/\d+")


def normalize_path(path: str) -> str:
    """
    Replace UUIDs and numeric IDs in the path with placeholders.

    ``/api/v1/documents/42`` → ``/api/v1/documents/:id``
    ``/api/v1/tenants/abc-uuid/…`` → ``/api/v1/tenants/:id/…``
    """
    path = _UUID_RE.sub(":id", path)
    path = _INT_ID_RE.sub("/:id", path)
    return path


# ── Helper: record DB pool stats ─────────────────────────────────────

def record_db_pool_stats() -> None:
    """Snapshot the SA engine pool stats into Prometheus gauges."""
    if not settings.METRICS_ENABLED:
        return
    try:
        from app.db.session import engine

        pool = engine.pool
        DB_POOL_SIZE.set(pool.size())
        DB_POOL_CHECKED_IN.set(pool.checkedin())
        DB_POOL_CHECKED_OUT.set(pool.checkedout())
    except Exception:
        pass  # best-effort


# ── Middleware ────────────────────────────────────────────────────────

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Record HTTP request count and duration.

    Skips instrumentation when ``METRICS_ENABLED`` is False.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.METRICS_ENABLED:
            return await call_next(request)

        method = request.method
        path = normalize_path(request.url.path)

        start = time.monotonic()
        try:
            response = await call_next(request)
        except Exception:
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status="500").inc()
            raise
        else:
            duration = time.monotonic() - start
            status = str(response.status_code)
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
            HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(duration)

            # Piggyback DB pool stats on every request (cheap)
            record_db_pool_stats()

            return response


# ── Helpers for hooks ─────────────────────────────────────────────────

def observe_query(
    *,
    tenant_id: str,
    mode: str,
    status: str,
    duration_s: float,
    tokens_query: int = 0,
    tokens_context: int = 0,
    tokens_total: int = 0,
) -> None:
    """Record AI query metrics (best-effort, non-blocking)."""
    if not settings.METRICS_ENABLED:
        return
    try:
        AI_QUERIES_TOTAL.labels(tenant_id=tenant_id, mode=mode, status=status).inc()
        AI_QUERY_LATENCY.labels(tenant_id=tenant_id, mode=mode).observe(duration_s)
        if tokens_query:
            AI_TOKENS_USED_TOTAL.labels(tenant_id=tenant_id, type="query").inc(tokens_query)
        if tokens_context:
            AI_TOKENS_USED_TOTAL.labels(tenant_id=tenant_id, type="context").inc(tokens_context)
        if tokens_total:
            AI_TOKENS_USED_TOTAL.labels(tenant_id=tenant_id, type="total").inc(tokens_total)
    except Exception:
        logger.debug("metrics.observe_query failed", exc_info=True)


def observe_rate_limit_hit(*, tenant_id: str, scope: str = "tenant") -> None:
    """Increment rate-limit hit counter (best-effort)."""
    if not settings.METRICS_ENABLED:
        return
    try:
        RATE_LIMIT_HITS_TOTAL.labels(tenant_id=tenant_id, scope=scope).inc()
    except Exception:
        logger.debug("metrics.rate_limit_hit failed", exc_info=True)


def observe_token_quota_exceeded(*, tenant_id: str) -> None:
    """Increment token-quota exceeded counter (best-effort)."""
    if not settings.METRICS_ENABLED:
        return
    try:
        TOKEN_QUOTA_EXCEEDED_TOTAL.labels(tenant_id=tenant_id).inc()
    except Exception:
        logger.debug("metrics.token_quota_exceeded failed", exc_info=True)


def observe_vector_query_latency(duration_s: float) -> None:
    """Record vector index query latency (best-effort)."""
    if not settings.METRICS_ENABLED:
        return
    try:
        VECTOR_QUERY_LATENCY.observe(duration_s)
    except Exception:
        logger.debug("metrics.vector_query_latency failed", exc_info=True)


# ── /metrics endpoint helper ──────────────────────────────────────────

def generate_metrics_response() -> bytes:
    """Return Prometheus-compatible metrics output."""
    return generate_latest(REGISTRY)
