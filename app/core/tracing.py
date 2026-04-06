"""
Distributed tracing (Phase 7.0) — OpenTelemetry integration.

Rules:
  - One trace per request, reuse request_id as trace context
  - Async, non-blocking, sampling-aware
  - Spans MUST NOT include: query text, prompts, embeddings, chunks,
    documents, snippets
  - Allowed attributes: tenant_id, user_id (optional), latency,
    result_count, mode, status
  - Tracing is failure-isolated: errors never propagate to callers
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Lazy initialization ──────────────────────────────────────────────

_tracer = None
_initialized = False


def _ensure_tracer():
    """Lazily initialize the OpenTelemetry tracer."""
    global _tracer, _initialized
    if _initialized:
        return _tracer
    _initialized = True

    if not settings.TRACING_ENABLED:
        _tracer = None
        return _tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

        sampler = TraceIdRatioBased(settings.TRACING_SAMPLE_RATE)
        provider = TracerProvider(sampler=sampler)

        # Optional exporter setup
        if settings.TRACING_EXPORTER == "console":
            from opentelemetry.sdk.trace.export import (
                SimpleSpanProcessor,
                ConsoleSpanExporter,
            )
            provider.add_span_processor(
                SimpleSpanProcessor(ConsoleSpanExporter())
            )
        elif settings.TRACING_EXPORTER == "otlp" and settings.OTEL_EXPORTER_OTLP_ENDPOINT:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = OTLPSpanExporter(
                    endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
                )
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except ImportError:
                logger.warning("tracing: OTLP exporter not installed, skipping")

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("ai-server", "7.0.0")

    except Exception:
        logger.warning("tracing: initialization failed", exc_info=True)
        _tracer = None

    return _tracer


def get_tracer():
    """Return the global tracer (or None if tracing is disabled)."""
    return _ensure_tracer()


# ── Span helpers ─────────────────────────────────────────────────────

@contextmanager
def traced_span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Generator:
    """
    Context manager that creates an OTel span with safe attributes.

    If tracing is disabled or fails, yields a no-op context.
    Privacy: only allowed attributes are set.
    """
    if not settings.TRACING_ENABLED:
        yield None
        return

    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    safe_attrs = _sanitize_span_attributes(attributes or {})

    try:
        with tracer.start_as_current_span(name, attributes=safe_attrs) as span:
            yield span
    except Exception:
        logger.debug("tracing.span_failed name=%s", name, exc_info=True)
        yield None


# ── Attribute sanitization ────────────────────────────────────────────

_ALLOWED_SPAN_ATTRS = frozenset({
    "tenant_id",
    "user_id",
    "latency",
    "latency_ms",
    "result_count",
    "mode",
    "status",
    "scope",
    "http.method",
    "http.status_code",
    "http.route",
    "span.kind",
})

_DENIED_SPAN_ATTRS = frozenset({
    "query",
    "query_text",
    "prompt",
    "text",
    "content",
    "snippet",
    "chunks",
    "documents",
    "embedding",
    "embeddings",
    "vector",
    "context",
    "passage",
    "raw",
    "message",
})


def _sanitize_span_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    """Return only allowed attributes; drop denied and unknown keys."""
    safe: dict[str, Any] = {}
    for k, v in attrs.items():
        if k in _DENIED_SPAN_ATTRS:
            continue
        if k in _ALLOWED_SPAN_ATTRS:
            # OTel attributes must be str/int/float/bool or sequences thereof
            if isinstance(v, (str, int, float, bool)):
                safe[k] = v
            elif v is not None:
                safe[k] = str(v)
    return safe


# ── Convenience helpers for common spans ──────────────────────────────

@contextmanager
def trace_http_request(
    *, method: str, route: str, request_id: str | None = None,
) -> Generator:
    """Trace an HTTP request span."""
    attrs = {
        "http.method": method,
        "http.route": route,
    }
    with traced_span("http.request", attributes=attrs) as span:
        yield span


@contextmanager
def trace_auth_check(*, tenant_id: str = "", user_id: int | None = None) -> Generator:
    """Trace authentication/authorization check."""
    attrs: dict[str, Any] = {"tenant_id": tenant_id}
    if user_id is not None:
        attrs["user_id"] = user_id
    with traced_span("auth.check", attributes=attrs) as span:
        yield span


@contextmanager
def trace_quota_check(*, tenant_id: str) -> Generator:
    """Trace quota enforcement check."""
    with traced_span("quota.check", attributes={"tenant_id": tenant_id}) as span:
        yield span


@contextmanager
def trace_rate_limit_check(*, tenant_id: str, scope: str = "tenant") -> Generator:
    """Trace rate-limit check."""
    with traced_span(
        "rate_limit.check",
        attributes={"tenant_id": tenant_id, "scope": scope},
    ) as span:
        yield span


@contextmanager
def trace_retrieval(
    *, tenant_id: str, mode: str = "hybrid",
) -> Generator:
    """Trace the retrieval (vector / bm25 / hybrid) step."""
    with traced_span(
        "retrieval",
        attributes={"tenant_id": tenant_id, "mode": mode},
    ) as span:
        yield span


@contextmanager
def trace_db_access(*, operation: str = "query") -> Generator:
    """Trace a database access span."""
    with traced_span("db.access", attributes={"status": operation}) as span:
        yield span


# ── Reset helper (for testing) ────────────────────────────────────────

def reset_tracer() -> None:
    """Reset the global tracer state (for tests only)."""
    global _tracer, _initialized
    _tracer = None
    _initialized = False
