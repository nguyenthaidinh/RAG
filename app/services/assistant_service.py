"""
Assistant service — orchestrates retrieval + answer generation.

Phase 2B: source-aware routing.
  - KNOWLEDGE → retrieval-first (legacy behavior)
  - SYSTEM    → system context only, retrieval skipped
  - ACCESS    → permissions context only, retrieval skipped
  - MIXED     → retrieval + system context combined
  - UNKNOWN   → knowledge fallback (legacy)

Fail-open rules:
  - Orchestrator failure → default to knowledge-first (retrieval always runs)
  - Retrieval is ONLY skipped when orchestration succeeds AND routing
    explicitly excludes knowledge (SYSTEM/ACCESS pure categories)
  - LLM failure → deterministic fallback via SystemAnswerBuilder or static msg
  - System context failure → knowledge path continues if applicable

Design rules:
  - No business logic specific to any consumer (Core-Platform, etc.)
  - No hard-coded URLs or routes
  - tenant_id always from auth, never from client
  - No fake citations for system context answers
  - Public API / response schema unchanged
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.schemas.assistant import (
    AssistantCitation,
    AssistantRespondRequest,
    AssistantRespondResponse,
)
from app.services.answer_service import (
    AnswerEvidence,
    AnswerService,
    AnswerSnippet,
    GeneratedAnswer,
)
from app.services.retrieval.factories import get_query_service

if TYPE_CHECKING:
    from app.services.retrieval.types import QueryResult

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Lazy singletons
# ──────────────────────────────────────────────────────────────────────────────

_query_svc = None


def _get_query_svc():
    global _query_svc
    if _query_svc is None:
        _query_svc = get_query_service()
    return _query_svc


_answer_svc = None


def _get_answer_svc():
    global _answer_svc
    if _answer_svc is None:
        _answer_svc = AnswerService()
    return _answer_svc


def _get_orchestrator():
    """System context orchestrator removed in CTDT fork — always returns None."""
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Fallback message templates
# ──────────────────────────────────────────────────────────────────────────────

_MSG_HAS_RESULTS = (
    "I found some relevant content for your question. "
    "Please see the citations below for details."
)
_MSG_NO_RESULTS = (
    "I could not find relevant content in the current knowledge base "
    "for your question."
)
_MSG_SYSTEM_NO_DATA = (
    "I don't have enough system information at the moment "
    "to answer this question. Please try again later or "
    "contact your administrator."
)
_MSG_ACCESS_NO_DATA = (
    "I don't have enough permission information at the moment "
    "to answer this question about access rights. "
    "Please contact your administrator for details."
)

# Max snippet length in citations to avoid oversized responses
_CITATION_SNIPPET_MAX_LEN = 500


# ──────────────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────────────


class AssistantService:
    """
    Orchestrates retrieval + answer generation for assistant responses.
    CTDT fork: knowledge-only path (system context removed).
    """

    __slots__ = ()

    async def respond(
        self,
        *,
        request: AssistantRespondRequest,
        tenant_id: str,
        user_id: int,
        trace_id: str | None = None,
    ) -> AssistantRespondResponse:
        # ── Resolve retrieval parameters ─────────────────────────────────
        retrieval = request.retrieval
        mode = "hybrid"
        final_limit = None
        vector_limit = None
        bm25_limit = None

        if retrieval is not None:
            if retrieval.mode is not None:
                mode = retrieval.mode
            final_limit = retrieval.final_limit or retrieval.top_k
            vector_limit = retrieval.vector_limit
            bm25_limit = retrieval.bm25_limit

        # ── Retrieval (always knowledge-first) ────────────────────────────
        results: list[QueryResult] = []
        query_svc = _get_query_svc()
        results = await query_svc.query(
            tenant_id=tenant_id,
            user_id=user_id,
            query_text=request.message,
            mode=mode,
            final_limit=final_limit,
            vector_limit=vector_limit,
            bm25_limit=bm25_limit,
            history=request.history or [],
        )

        snippets = _map_results_to_snippets(results)

        # ── Generate structured answer (fail-open) ─────────────────────
        generated: GeneratedAnswer | None = None
        try:
            answer_svc = _get_answer_svc()
            generated = await answer_svc.generate_structured(
                question=request.message,
                snippets=snippets,
                history=request.history or [],
                system_context_block="",
                question_category="knowledge",
                allow_system_context_only=False,
            )
        except Exception:
            logger.warning(
                "assistant.answer_generation_failed tenant_id=%s user_id=%d",
                tenant_id, user_id, exc_info=True,
            )

        # ── Build message + citations ─────────────────────────────────────
        message: str
        citations: list[AssistantCitation]

        if generated and generated.text:
            message = generated.text
            citations = _build_citations_from_evidences(generated.evidences)
        elif generated and generated.evidences:
            message = _MSG_HAS_RESULTS
            citations = _build_citations_from_evidences(generated.evidences)
        else:
            citations = _build_citations(results)
            message = _MSG_HAS_RESULTS if citations else _MSG_NO_RESULTS

        logger.info(
            "assistant.respond tenant_id=%s user_id=%d citations=%d",
            tenant_id, user_id, len(citations),
        )

        return AssistantRespondResponse(
            message=message,
            citations=citations,
            actions=[],
            trace_id=trace_id,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _category_no_data_message(category: str) -> str:
    """Return an appropriate no-data message for the category."""
    if category == "access":
        return _MSG_ACCESS_NO_DATA
    if category == "system":
        return _MSG_SYSTEM_NO_DATA
    return _MSG_NO_RESULTS


def _map_results_to_snippets(results: list[QueryResult]) -> list[AnswerSnippet]:
    """Map retrieval QueryResult items → AnswerSnippet for answer generation."""
    return [
        AnswerSnippet(
            document_id=r.document_id,
            chunk_id=r.chunk_id,
            snippet=r.snippet,
            score=r.score,
            source_document_id=getattr(r, "source_document_id", None),
            title=getattr(r, "title", None),
            heading=getattr(r, "heading", None),
            debug_meta=getattr(r, "debug_meta", None),
        )
        for r in results
    ]


def _build_citations_from_evidences(
    evidences: tuple[AnswerEvidence, ...],
) -> list[AssistantCitation]:
    """Map selected AnswerEvidence → AssistantCitation."""
    citations: list[AssistantCitation] = []

    for e in evidences:
        snippet = e.snippet or ""
        if len(snippet) > _CITATION_SNIPPET_MAX_LEN:
            snippet = snippet[:_CITATION_SNIPPET_MAX_LEN] + "…"

        citations.append(
            AssistantCitation(
                chunk_id=e.chunk_id,
                document_id=e.document_id,
                source_document_id=e.source_document_id,
                title=e.title,
                heading=e.heading,
                snippet=snippet or None,
                score=e.score,
                rank=e.rank,
                metadata=None,
            )
        )

    return citations


def _build_citations(results: list[QueryResult]) -> list[AssistantCitation]:
    """Fallback: map ALL retrieval results → citations.

    Used only when generate_structured() fails entirely (exception).
    """
    citations: list[AssistantCitation] = []

    for r in results:
        snippet = getattr(r, "snippet", None) or ""
        if len(snippet) > _CITATION_SNIPPET_MAX_LEN:
            snippet = snippet[:_CITATION_SNIPPET_MAX_LEN] + "…"

        citations.append(
            AssistantCitation(
                chunk_id=getattr(r, "chunk_id", None),
                document_id=getattr(r, "document_id", None),
                source_document_id=getattr(r, "source_document_id", None),
                title=None,
                snippet=snippet or None,
                score=getattr(r, "score", None),
                metadata=None,
            )
        )

    return citations
