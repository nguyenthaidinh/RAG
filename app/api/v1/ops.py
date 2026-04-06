"""
Admin observability dashboard API (Phase 8.0).

STRICTLY READ-ONLY ops view for incident debugging, system visibility,
and capacity awareness.

ALLOWED data: request counts, error rates, latency percentiles,
  queue depth, top tenants by usage.

FORBIDDEN (ABSOLUTE): raw queries, prompts, chunks, embeddings,
  documents. No mutations. No admin actions. No business logic.

Gated behind OPS_DASHBOARD_ENABLED flag.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends
from starlette.responses import JSONResponse

from app.core.auth_deps import require_admin
from app.core.config import settings
from app.db.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ops", tags=["ops"])


def _safe_get_metrics() -> dict:
    """Collect Prometheus metrics summary (best-effort)."""
    result = {
        "http_requests_total": 0,
        "http_errors_5xx": 0,
        "ai_queries_total": 0,
        "rate_limit_hits_total": 0,
        "token_quota_exceeded_total": 0,
    }
    try:
        from app.core.metrics import (
            HTTP_REQUESTS_TOTAL,
            AI_QUERIES_TOTAL,
            RATE_LIMIT_HITS_TOTAL,
            TOKEN_QUOTA_EXCEEDED_TOTAL,
        )

        # Sum all HTTP requests
        for sample in HTTP_REQUESTS_TOTAL.collect()[0].samples:
            result["http_requests_total"] += int(sample.value)
            if sample.labels.get("status", "").startswith("5"):
                result["http_errors_5xx"] += int(sample.value)

        # Sum AI queries
        for sample in AI_QUERIES_TOTAL.collect()[0].samples:
            result["ai_queries_total"] += int(sample.value)

        # Sum rate limit hits
        for sample in RATE_LIMIT_HITS_TOTAL.collect()[0].samples:
            result["rate_limit_hits_total"] += int(sample.value)

        # Sum quota exceeded
        for sample in TOKEN_QUOTA_EXCEEDED_TOTAL.collect()[0].samples:
            result["token_quota_exceeded_total"] += int(sample.value)

    except Exception:
        logger.debug("ops.metrics_collection_failed", exc_info=True)

    return result


def _safe_get_top_tenants() -> list[dict]:
    """Extract top tenants by query count from Prometheus metrics."""
    tenants: dict[str, int] = {}
    try:
        from app.core.metrics import AI_QUERIES_TOTAL

        for sample in AI_QUERIES_TOTAL.collect()[0].samples:
            tid = sample.labels.get("tenant_id", "")
            if tid:
                tenants[tid] = tenants.get(tid, 0) + int(sample.value)
    except Exception:
        pass

    # Sort descending, return top 10
    sorted_tenants = sorted(tenants.items(), key=lambda x: x[1], reverse=True)[:10]
    return [{"tenant_id": tid, "queries": count} for tid, count in sorted_tenants]


def _safe_get_latency_stats() -> dict:
    """Extract latency statistics from histograms (best-effort)."""
    result = {
        "http_p50_ms": None,
        "http_p95_ms": None,
        "http_p99_ms": None,
    }
    try:
        from app.core.metrics import HTTP_REQUEST_DURATION

        # Collect histogram data
        families = HTTP_REQUEST_DURATION.collect()
        if families:
            buckets: list[tuple[float, float]] = []
            count_total = 0.0
            for sample in families[0].samples:
                if sample.name.endswith("_bucket"):
                    buckets.append((float(sample.labels.get("le", "inf")), sample.value))
                elif sample.name.endswith("_count"):
                    count_total += sample.value

            if count_total > 0 and buckets:
                buckets.sort(key=lambda x: x[0])
                for pct, key in [(0.50, "http_p50_ms"), (0.95, "http_p95_ms"), (0.99, "http_p99_ms")]:
                    target = count_total * pct
                    for bound, cum in buckets:
                        if cum >= target:
                            result[key] = round(bound * 1000, 1)
                            break
    except Exception:
        pass

    return result


def _safe_get_backpressure_stats() -> dict:
    """Get current backpressure statistics."""
    try:
        from app.core.backpressure import get_backpressure_manager
        return get_backpressure_manager().stats()
    except Exception:
        return {}


def _safe_get_shutdown_stats() -> dict:
    """Get current shutdown/in-flight stats."""
    try:
        from app.core.shutdown import get_shutdown_manager
        mgr = get_shutdown_manager()
        return {
            "in_flight": mgr.in_flight,
            "is_draining": mgr.is_draining,
            "uptime_seconds": round(mgr.uptime_seconds, 1),
        }
    except Exception:
        return {}


def _safe_get_db_pool_stats() -> dict:
    """Get DB connection pool statistics."""
    try:
        from app.db.session import engine
        pool = engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }
    except Exception:
        return {}


@router.get("/dashboard")
async def ops_dashboard(
    _admin: User = Depends(require_admin),
) -> JSONResponse:
    """
    Read-only ops dashboard — aggregated system metrics.

    Returns counts, rates, latencies, queue depth, and top tenants.
    NEVER returns raw queries, prompts, chunks, or documents.
    """
    if not settings.OPS_DASHBOARD_ENABLED:
        return JSONResponse(
            {"status": "disabled"},
            status_code=200,
        )

    metrics = _safe_get_metrics()
    top_tenants = _safe_get_top_tenants()
    latency = _safe_get_latency_stats()
    backpressure = _safe_get_backpressure_stats()
    shutdown_info = _safe_get_shutdown_stats()
    db_pool = _safe_get_db_pool_stats()

    # Compute error rate
    error_rate = 0.0
    if metrics["http_requests_total"] > 0:
        error_rate = round(
            metrics["http_errors_5xx"] / metrics["http_requests_total"] * 100, 2
        )

    return JSONResponse({
        "status": "ok",
        "request_counts": {
            "http_total": metrics["http_requests_total"],
            "http_errors_5xx": metrics["http_errors_5xx"],
            "ai_queries_total": metrics["ai_queries_total"],
            "rate_limit_hits": metrics["rate_limit_hits_total"],
            "token_quota_exceeded": metrics["token_quota_exceeded_total"],
        },
        "error_rate_pct": error_rate,
        "latency": latency,
        "top_tenants": top_tenants,
        "backpressure": backpressure,
        "inflight": shutdown_info,
        "db_pool": db_pool,
    })
