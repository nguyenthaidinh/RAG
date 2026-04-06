"""
System context debug endpoint (Phase 1.1 — Hardened).

Dev/internal-only endpoint that exercises the full Phase 1 pipeline:
  question classifier → connector → context builder → orchestration

Phase 1.1 hardening:
  - Gated by SYSTEM_CONTEXT_DEBUG_ENABLED setting.
  - Role-gated: requires admin role (system_admin or tenant_admin).
  - Provider override validated against registry — rejects unknown providers.
  - Removed unused db dependency.
  - Response sanitized: no raw PII leak.

NOT for production traffic — no billing, no quota checks.

Provides:
  POST /api/v1/system-context/debug
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.auth_deps import require_admin
from app.core.config import settings
from app.db.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/system-context", tags=["system-context-debug"])


# ── Request / Response schemas ───────────────────────────────────────


class SystemContextDebugRequest(BaseModel):
    """Request body for the debug endpoint."""

    question: str = Field(..., min_length=1, max_length=2000)
    build_context: bool = Field(
        default=True,
        description="If True, build system context bundle via connector.",
    )
    provider: str | None = Field(
        default=None,
        description="Connector provider override. Must be a registered provider.",
    )


class SystemContextDebugResponse(BaseModel):
    """Response from the debug endpoint."""

    classification: dict
    routing: dict
    context_bundle: dict | None = None
    provider_used: str | None = None
    notes: list[str] = Field(default_factory=list)


# ── Endpoint ─────────────────────────────────────────────────────────


@router.post("/debug", response_model=SystemContextDebugResponse)
async def system_context_debug(
    body: SystemContextDebugRequest,
    user: User = Depends(require_admin),
):
    """
    Debug endpoint: classify question + build system context.

    Requires admin role.  Gated by SYSTEM_CONTEXT_DEBUG_ENABLED.
    """
    if not settings.SYSTEM_CONTEXT_DEBUG_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    tenant_id = user.tenant_id
    user_id = user.id

    # ── Resolve connector ────────────────────────────────────────
    from app.services.system_context.connector_registry import get_connector_registry
    from app.services.system_context.context_builder import SystemContextBuilder
    from app.services.orchestration.system_context_orchestrator import (
        SystemContextOrchestrator,
    )

    registry = get_connector_registry()
    provider_name = body.provider or settings.SYSTEM_CONTEXT_PROVIDER

    # Validate provider override against registry
    if body.provider is not None and body.provider not in registry:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown provider '{body.provider}'. "
                f"Available: {registry.list_providers()}"
            ),
        )

    connector = registry.get(provider_name)
    if not connector:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Configured provider '{provider_name}' not available. "
                f"Available: {registry.list_providers()}"
            ),
        )

    # ── Build orchestrator with connector ─────────────────────────
    builder = SystemContextBuilder(connector=connector)
    orchestrator = SystemContextOrchestrator(context_builder=builder)

    # ── Evaluate ─────────────────────────────────────────────────
    try:
        result = await orchestrator.evaluate(
            question=body.question,
            tenant_id=tenant_id,
            actor_user_id=user_id,
            build_context=body.build_context,
        )
    except Exception:
        logger.error(
            "system_context_debug.failed tenant_id=%s",
            tenant_id, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="System context evaluation failed",
        )

    # ── Build response (sanitized) ────────────────────────────────
    bundle_dict = None
    context_sections: list[str] = []
    rendered_block_chars = 0

    if result.context_bundle is not None:
        bundle_dict = result.context_bundle.model_dump(mode="json")
        b = result.context_bundle
        if b.user is not None:
            context_sections.append("user")
        if b.tenant is not None:
            context_sections.append("tenant")
        if b.has_permissions:
            context_sections.append("permissions")
        if b.has_stats:
            context_sections.append("stats")
        if b.records:
            context_sections.append(f"records({len(b.records)})")
        if b.workflows:
            context_sections.append(f"workflows({len(b.workflows)})")

        # Render to measure block size
        from app.services.system_context.context_renderer import (
            render_system_context_block,
        )
        rendered_block_chars = len(render_system_context_block(b))

    return SystemContextDebugResponse(
        classification=result.classification.telemetry_dict(),
        routing={
            "category": result.category.value,
            "should_use_knowledge": result.should_use_knowledge,
            "should_use_system_context": result.should_use_system_context,
            "should_use_access_context": result.should_use_access_context,
            "context_sections": context_sections,
            "rendered_block_chars": rendered_block_chars,
        },
        context_bundle=bundle_dict,
        provider_used=provider_name,
        notes=list(result.notes),
    )
