"""
Usage logging middleware.

Enqueues a lightweight DTO to the shared UsageLogDispatcher.
No per-request DB session. No asyncio.create_task per request.
Fail-open: enqueue errors never break the request path.
Privacy-safe: never logs raw payloads, tokens, query text.
"""
import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.usage_dispatcher import UsageLogEntry, get_usage_dispatcher

logger = logging.getLogger(__name__)

_SKIP_PREFIXES = ("/docs", "/openapi.json", "/redoc", "/health", "/metrics", "/static")


class UsageLoggingMiddleware(BaseHTTPMiddleware):
    """
    Capture usage metadata after response, enqueue to dispatcher.

    - Fail-open: enqueue errors are swallowed
    - No DB session per request
    - No unbounded task creation
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if any(path.startswith(p) for p in _SKIP_PREFIXES):
            return await call_next(request)

        response = await call_next(request)

        # Only log authenticated requests with a known tenant
        user_id = getattr(request.state, "user_id", None)
        tenant_id = getattr(request.state, "tenant_id", None)
        if not user_id or not tenant_id:
            return response

        try:
            # Derive request_id: prefer state (set by RequestIdMiddleware),
            # fallback to header, fallback to generated uuid.
            request_id = (
                getattr(request.state, "request_id", None)
                or request.headers.get("X-Request-ID")
                or str(uuid.uuid4())
            )

            entry = UsageLogEntry(
                request_id=request_id,
                user_id=user_id,
                tenant_id=tenant_id,
                endpoint=path,
                method=request.method,
                status_code=response.status_code,
                success=200 <= response.status_code < 400,
                api_key_id=getattr(request.state, "api_key_id", None),
                tokens_input=getattr(request.state, "tokens_input", 0),
                tokens_output=getattr(request.state, "tokens_output", 0),
                tokens_total=getattr(request.state, "tokens_total", 0),
                file_size_bytes=getattr(request.state, "file_size_bytes", 0),
                request_cost=getattr(request.state, "request_cost", 0.0),
            )
            get_usage_dispatcher().enqueue(entry)
        except Exception:
            # Fail-open: never break request path
            logger.error("usage.enqueue_failed", exc_info=True)

        return response
