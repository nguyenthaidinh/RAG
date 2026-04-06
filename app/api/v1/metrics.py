"""
Prometheus metrics endpoint (Phase 7.0).

GET /metrics — admin/internal only, not publicly accessible.
Returns Prometheus-compatible text when METRICS_ENABLED is True.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.responses import Response

from app.core.auth_deps import require_admin
from app.core.config import settings
from app.db.models.user import User

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics_endpoint(
    _admin: User = Depends(require_admin),
) -> Response:
    """
    Expose Prometheus metrics (admin-only).

    Returns 200 with text/plain body when metrics are enabled.
    Returns 200 with empty body when metrics are disabled.
    """
    if not settings.METRICS_ENABLED:
        return Response(
            content="# metrics disabled\n",
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    from app.core.metrics import generate_metrics_response

    body = generate_metrics_response()
    return Response(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
