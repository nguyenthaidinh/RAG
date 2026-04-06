"""
Low-level retrieval / search / debug API endpoint.

This is the RETRIEVAL layer — it returns ranked document chunks with
scores and highlights.  It does NOT generate answers or chatbot responses.
For chatbot/answer functionality, use /api/v1/assistant/respond instead.

Exposes the retrieval pipeline with quota & rate-limit enforcement.
After successful execution, emits a QUERY_EXECUTED audit event.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import get_current_user
from app.core.config import settings
from app.db.models.user import User
from app.db.session import get_db
from app.deps.idempotency import get_idempotency_key
from app.repos.query_usage_repo import InMemoryQueryUsageRepository, PgQueryUsageRepository
from app.repos.rate_limit_repo import RateLimitRepository
from app.services.quota_enforcement_service import QuotaEnforcementService
from app.services.quota_policy_service import QuotaPolicyService
from app.services.rate_limit_service import RateLimitService
from app.services.token_ledger import get_token_ledger
from app.services.token_quota_service import TokenQuotaService
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

    # Step 5: debug mode — when true, response includes selection metadata
    include_debug: bool = False

    # Conversation history for follow-up context (V1)
    history: list[ConversationTurn] = Field(default_factory=list)

    @field_validator("mode", mode="before")
    @classmethod
    def normalize_mode(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @model_validator(mode="after")
    def _normalize_legacy_fields(self) -> "QueryRequest":
        # ── query / query_text ───────────────────────────────────
        if self.query is None:
            if self.query_text is not None:
                self.query = self.query_text
            else:
                raise ValueError("Either 'query' or 'query_text' must be provided")

        # ── final_limit / top_k ──────────────────────────────────
        if self.final_limit is None and self.top_k is not None:
            self.final_limit = self.top_k

        return self


class QueryResultItem(BaseModel):
    chunk_id: int
    document_id: int
    score: float
    snippet: str
    highlights: list[str]

    # Step 5: source fidelity — original doc for citation
    source_document_id: int | None = None

    # Step 5: debug metadata — only present when include_debug=true
    debug_meta: dict | None = None


class QueryResponse(BaseModel):
    # NEW: RAG synthesis (best-effort). If disabled/fails -> null.
    answer: str | None = None

    results: list[QueryResultItem]
    count: int


# ──────────────────────────────────────────────────────────────────────────────
# Lazy singletons
# ──────────────────────────────────────────────────────────────────────────────

_enforcement_svc: QuotaEnforcementService | None = None
_query_svc = None


def _get_enforcement_svc() -> QuotaEnforcementService:
    global _enforcement_svc

    if _enforcement_svc is None:
        try:
            usage_repo = PgQueryUsageRepository()
        except Exception:
            usage_repo = InMemoryQueryUsageRepository()

        _enforcement_svc = QuotaEnforcementService(
            policy_svc=QuotaPolicyService(),
            rate_limit_svc=RateLimitService(RateLimitRepository()),
            token_quota_svc=TokenQuotaService(get_token_ledger()),
            usage_repo=usage_repo,
        )

    return _enforcement_svc


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
    response: Response,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    idempotency_key: str = Depends(get_idempotency_key),
):
    """
    Execute a retrieval query with quota & rate-limit enforcement.

    Headers:
      X-Idempotency-Key: client-provided idempotency key (optional)
    """
    t0 = time.monotonic()

    tenant_id = user.tenant_id
    user_id = user.id
    request_id = getattr(request.state, "request_id", None)

    # ── Conservative token preflight estimate ────────────────────────────────
    final_limit = body.final_limit or settings.QUERY_FINAL_LIMIT

    # Approx: ~4 chars per token (best-effort)
    query_tokens_est = max(1, len(body.query) // 4)
    tokens_estimate = query_tokens_est + (final_limit * settings.TOKEN_QUOTA_CONTEXT_ESTIMATE)

    # ── Quota enforcement BEFORE expensive retrieval ─────────────────────────
    enforcement = _get_enforcement_svc()

    rl_result = await enforcement.enforce_or_raise(
        db,
        tenant_id=tenant_id,
        user_id=user_id,
        idempotency_key=idempotency_key,
        tokens_estimate=tokens_estimate,
        request_id=request_id,
    )

    # Attach rate-limit headers
    if rl_result.limit > 0:
        for k, v in rl_result.headers().items():
            response.headers[k] = v

    # ── Execute retrieval pipeline ───────────────────────────────────────────
    query_svc = _get_query_svc()

    results = await query_svc.query(
        tenant_id=tenant_id,
        user_id=user_id,
        query_text=body.query,
        idempotency_key=idempotency_key,
        final_limit=final_limit,
        vector_limit=body.vector_limit,
        bm25_limit=body.bm25_limit,
        mode=body.mode,  # ✅ PASS MODE END-TO-END
        include_debug=body.include_debug,
        history=body.history or [],  # Phase 3A: pass history for rewrite
    )

    # ── Timing ───────────────────────────────────────────────────────────────
    dt = time.monotonic() - t0
    elapsed_ms = int(dt * 1000)
    elapsed_s = dt

    # ── Phase 6.0: emit QUERY_EXECUTED audit event ───────────────────────────
    try:
        from app.services.audit_service import get_audit_service

        await get_audit_service().log_query_executed(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            results_count=len(results),
            latency_ms=elapsed_ms,
            mode=body.mode,
        )
    except Exception:
        # Audit failures MUST NOT break the query response
        logger.warning(
            "audit.query_executed_failed tenant_id=%s",
            tenant_id,
            exc_info=True,
        )

    # ── Phase 7.0: observability hooks (best-effort) ─────────────────────────
    try:
        from app.core.metrics import observe_query

        observe_query(
            tenant_id=tenant_id,
            mode=body.mode,
            status="success",
            duration_s=elapsed_s,
        )
    except Exception:
        # Observability must never break user response
        pass

    # ── NEW: best-effort answer synthesis (fail-open) ────────────────────────
    answer: str | None = None
    try:
        from app.services.answer_service import AnswerService, AnswerSnippet

        # Build small snippets list from top results (no highlights needed)
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
        # MUST NOT break response
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