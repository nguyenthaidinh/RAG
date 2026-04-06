"""
Low-level retrieval / search / debug API endpoint.

Returns ranked document chunks with scores and highlights.
For chatbot/answer functionality, use /api/v1/assistant/respond instead.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import get_current_user
from app.core.config import settings
from app.db.models.user import User
from app.db.session import get_db
from app.services.retrieval.factories import get_query_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/query", tags=["query"])


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────


class ConversationTurn(BaseModel):
    """A single turn in the conversation history."""
    role: Literal["user", "assistant"]
    text: str = Field(max_length=2000)


class QueryRequest(BaseModel):
    query: str | None = Field(default=None, min_length=1, max_length=8000)
    query_text: str | None = Field(default=None, exclude=True, min_length=1, max_length=8000)

    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"

    vector_limit: int | None = Field(default=None, ge=1, le=500)
    bm25_limit: int | None = Field(default=None, ge=1, le=500)
    final_limit: int | None = Field(default=None, ge=1, le=100)
    top_k: int | None = Field(default=None, exclude=True, ge=1, le=100)

    include_debug: bool = False

    history: list[ConversationTurn] = Field(default_factory=list)

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @model_validator(mode="after")
    def _normalize_legacy_fields(self) -> "QueryRequest":
        if self.query is None:
            if self.query_text is not None:
                self.query = self.query_text
            else:
                raise ValueError("Either 'query' or 'query_text' must be provided")

        if self.final_limit is None and self.top_k is not None:
            self.final_limit = self.top_k

        return self


class QueryResultItem(BaseModel):
    chunk_id: int
    document_id: int
    score: float
    snippet: str
    highlights: list[str]
    source_document_id: int | None = None
    debug_meta: dict | None = None


class QueryResponse(BaseModel):
    answer: str | None = None
    results: list[QueryResultItem]
    count: int


# ──────────────────────────────────────────────────────────────────────────────
# Lazy singleton
# ──────────────────────────────────────────────────────────────────────────────

_query_svc = None


def _get_query_svc():
    global _query_svc
    if _query_svc is None:
        _query_svc = get_query_service()
    return _query_svc


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────────────────────────────────────


@router.post("", response_model=QueryResponse)
async def execute_query(
    body: QueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Execute a retrieval query."""
    t0 = time.monotonic()

    tenant_id = user.tenant_id
    user_id = user.id
    request_id = getattr(request.state, "request_id", None)

    final_limit = body.final_limit or settings.QUERY_FINAL_LIMIT

    # ── Execute retrieval pipeline ────────────────────────────────────────────
    query_svc = _get_query_svc()

    results = await query_svc.query(
        tenant_id=tenant_id,
        user_id=user_id,
        query_text=body.query,
        idempotency_key=None,
        final_limit=final_limit,
        vector_limit=body.vector_limit,
        bm25_limit=body.bm25_limit,
        mode=body.mode,
        include_debug=body.include_debug,
        history=body.history or [],
    )

    dt = time.monotonic() - t0
    elapsed_ms = int(dt * 1000)
    elapsed_s = dt

    # ── Audit event (best-effort) ─────────────────────────────────────────────
    try:
        from app.services.audit_service import get_audit_service
        await get_audit_service().log_query_executed(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            request_id=request_id,
            idempotency_key=None,
            results_count=len(results),
            latency_ms=elapsed_ms,
            mode=body.mode,
        )
    except Exception:
        logger.warning("audit.query_executed_failed tenant_id=%s", tenant_id, exc_info=True)

    # ── Metrics (best-effort) ─────────────────────────────────────────────────
    try:
        from app.core.metrics import observe_query
        observe_query(tenant_id=tenant_id, mode=body.mode, status="success", duration_s=elapsed_s)
    except Exception:
        pass

    # ── Answer synthesis (best-effort, fail-open) ─────────────────────────────
    answer: str | None = None
    try:
        from app.services.answer_service import AnswerService, AnswerSnippet
        max_results = int(getattr(settings, "LLM_ANSWER_MAX_RESULTS", 6))
        snippets = [
            AnswerSnippet(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                snippet=r.snippet,
                score=r.score,
            )
            for r in results[:max_results]
        ]
        answer = await AnswerService().generate(
            question=body.query,
            snippets=snippets,
            history=body.history or [],
        )
    except Exception:
        answer = None

    return QueryResponse(
        answer=answer,
        results=[
            QueryResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                score=r.score,
                snippet=r.snippet,
                highlights=list(r.highlights),
                source_document_id=getattr(r, "source_document_id", None),
                debug_meta=getattr(r, "debug_meta", None) if body.include_debug else None,
            )
            for r in results
        ],
        count=len(results),
    )
