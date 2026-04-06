"""
High-level assistant / chatbot answer API.

Provides:
  POST /api/v1/assistant/respond

This is the ANSWER layer — it runs retrieval internally, generates
LLM-powered answers, and returns structured responses with evidence-based
citations.  For raw retrieval/search/debug, use /api/v1/query instead.

This is a THIN router:
  - Validates the request
  - Extracts auth context (user, tenant) via existing deps
  - Delegates to AssistantService
  - Returns the response schema

No business logic lives here.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import get_current_user
from app.db.models.user import User
from app.db.session import get_db
from app.schemas.assistant import AssistantRespondRequest, AssistantRespondResponse
from app.services.assistant_service import AssistantService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])

# Singleton service instance (stateless — safe to share)
_assistant_svc = AssistantService()


@router.post("/respond", response_model=AssistantRespondResponse)
async def assistant_respond(
    body: AssistantRespondRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    High-level assistant endpoint.

    Accepts a user message, runs retrieval internally, and returns
    a structured assistant response with citations.

    Authentication: same as all other AI Server endpoints (JWT / API key).
    Tenant isolation: enforced via the authenticated user's tenant_id.
    """
    tenant_id = user.tenant_id
    user_id = user.id
    trace_id = getattr(request.state, "request_id", None)

    try:
        response = await _assistant_svc.respond(
            request=body,
            tenant_id=tenant_id,
            user_id=user_id,
            trace_id=trace_id,
        )
        return response

    except HTTPException:
        # Re-raise FastAPI HTTP exceptions as-is
        raise

    except Exception:
        logger.error(
            "assistant.respond_failed tenant_id=%s user_id=%d",
            tenant_id,
            user_id,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing your request.",
        )
