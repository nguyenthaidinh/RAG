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

from app.core.config import settings
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
    from app.services.orchestration.system_context_orchestrator import (
        OrchestrationResult,
        SystemContextOrchestrator,
    )
    from app.services.retrieval.types import QueryResult

from app.services.system_context.source_trace import AnswerSourceTrace

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


_orchestrator: SystemContextOrchestrator | None = None
_orchestrator_init_attempted: bool = False


def _get_orchestrator() -> SystemContextOrchestrator | None:
    """Lazy-init the system context orchestrator.

    Returns None when:
      - SYSTEM_CONTEXT_ENABLED is False
      - Provider not registered
      - Any init error (logged, fail-open)

    The init is attempted only once to avoid repeated errors.
    """
    global _orchestrator, _orchestrator_init_attempted

    if _orchestrator is not None:
        return _orchestrator

    if _orchestrator_init_attempted:
        return None

    _orchestrator_init_attempted = True

    if not getattr(settings, "SYSTEM_CONTEXT_ENABLED", False):
        return None

    try:
        from app.services.system_context.connector_registry import (
            get_connector_registry,
        )
        from app.services.system_context.context_builder import SystemContextBuilder
        from app.services.orchestration.system_context_orchestrator import (
            SystemContextOrchestrator,
        )

        provider_name = getattr(settings, "SYSTEM_CONTEXT_PROVIDER", "mock")
        registry = get_connector_registry()
        connector = registry.get(provider_name)

        if connector is None:
            logger.warning(
                "assistant.orchestrator_init provider=%s not_found "
                "available=%s — system context disabled",
                provider_name, registry.list_providers(),
            )
            return None

        builder = SystemContextBuilder(connector=connector)
        _orchestrator = SystemContextOrchestrator(context_builder=builder)

        logger.info(
            "assistant.orchestrator_init_ok provider=%s",
            provider_name,
        )
        return _orchestrator

    except Exception:
        logger.warning(
            "assistant.orchestrator_init_failed — system context disabled",
            exc_info=True,
        )
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

    Phase 2B: source-aware routing based on OrchestrationResult.
      - Respects should_use_knowledge / should_use_system_context /
        should_use_access_context routing flags
      - Skips retrieval for pure SYSTEM/ACCESS questions
      - Deterministic fallback via SystemAnswerBuilder when LLM unavailable
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

        # ── Orchestration (Phase 2B — fail-open) ─────────────────────────
        system_context_block = ""
        context_bundle = None
        orch_result: OrchestrationResult | None = None
        orchestration_ok = False

        # Default routing: knowledge-first (safe fallback)
        should_use_knowledge = True
        should_use_system_context = False
        should_use_access_context = False
        category_value = "knowledge"

        orchestrator = _get_orchestrator()
        if orchestrator is not None:
            try:
                orch_result = await orchestrator.evaluate(
                    question=request.message,
                    tenant_id=tenant_id,
                    actor_user_id=user_id,
                    build_context=True,
                )

                orchestration_ok = True
                category_value = orch_result.category.value
                should_use_knowledge = orch_result.should_use_knowledge
                should_use_system_context = orch_result.should_use_system_context
                should_use_access_context = orch_result.should_use_access_context
                context_bundle = orch_result.context_bundle

                if context_bundle is not None:
                    from app.services.system_context.context_renderer import (
                        render_system_context_block,
                    )
                    system_context_block = render_system_context_block(
                        context_bundle,
                    )

                logger.info(
                    "assistant.orchestration tenant_id=%s category=%s "
                    "knowledge=%s system=%s access=%s "
                    "has_bundle=%s ctx_chars=%d",
                    tenant_id,
                    category_value,
                    should_use_knowledge,
                    should_use_system_context,
                    should_use_access_context,
                    context_bundle is not None,
                    len(system_context_block),
                )

            except Exception:
                logger.warning(
                    "assistant.orchestration_failed tenant_id=%s "
                    "user_id=%d — falling back to knowledge-first",
                    tenant_id, user_id, exc_info=True,
                )
                # orchestration_ok stays False → knowledge-first fallback

        # ── Retrieval routing (Phase 2B) ─────────────────────────────────
        # Only skip retrieval when orchestration succeeded AND routing
        # explicitly excludes knowledge.
        retrieval_skipped = False
        results: list[QueryResult] = []

        if should_use_knowledge:
            # KNOWLEDGE / MIXED / UNKNOWN / fallback → always retrieve
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
        else:
            # SYSTEM / ACCESS → skip retrieval
            retrieval_skipped = True

        # ── Build response ───────────────────────────────────────────────
        snippets = _map_results_to_snippets(results)

        # ── Generate structured answer (fail-open) ─────────────────────
        allow_system_only = (
            should_use_system_context or should_use_access_context
        )

        generated: GeneratedAnswer | None = None
        try:
            answer_svc = _get_answer_svc()
            generated = await answer_svc.generate_structured(
                question=request.message,
                snippets=snippets,
                history=request.history or [],
                system_context_block=system_context_block,
                question_category=category_value,
                allow_system_context_only=allow_system_only,
            )
        except Exception:
            logger.warning(
                "assistant.answer_generation_failed tenant_id=%s user_id=%d "
                "category=%s",
                tenant_id, user_id, category_value, exc_info=True,
            )

        # ── Build message + citations with Phase 2B fallback cascade ────
        #
        # Precedence depends on category:
        #   SYSTEM / ACCESS: LLM → system_fallback → evidence_fallback → static
        #   MIXED:           LLM → evidence_fallback → system_fallback → static
        #   KNOWLEDGE / *:   LLM → evidence_fallback → static
        #
        # Rationale: MIXED questions have document grounding from retrieval.
        # When LLM fails, preserving document evidence + citations is more
        # valuable than a system-only answer that drops the citations.
        #

        message: str
        citations: list[AssistantCitation]
        answer_source: str

        if generated and generated.text:
            # Case A: LLM produced an answer — always best case
            message = generated.text
            citations = _build_citations_from_evidences(generated.evidences)
            answer_source = "llm"

        elif category_value == "mixed" and not (generated and generated.text):
            # Case B-MIXED: evidence_fallback BEFORE system_fallback
            if generated and generated.evidences:
                message = _MSG_HAS_RESULTS
                citations = _build_citations_from_evidences(generated.evidences)
                answer_source = "evidence_fallback"
            elif context_bundle is not None:
                from app.services.system_context.system_answer_builder import (
                    build_system_only_answer,
                )
                system_answer = build_system_only_answer(
                    question=request.message,
                    category=category_value,
                    bundle=context_bundle,
                )
                if system_answer:
                    message = system_answer
                    citations = []
                    answer_source = "system_fallback"
                else:
                    message = _category_no_data_message(category_value)
                    citations = []
                    answer_source = "deterministic"
            else:
                message = _MSG_NO_RESULTS
                citations = []
                answer_source = "deterministic"

        elif (
            allow_system_only
            and context_bundle is not None
            and not (generated and generated.text)
        ):
            # Case B-SYSTEM/ACCESS: system_fallback BEFORE evidence_fallback
            from app.services.system_context.system_answer_builder import (
                build_system_only_answer,
            )
            system_answer = build_system_only_answer(
                question=request.message,
                category=category_value,
                bundle=context_bundle,
            )

            if system_answer:
                message = system_answer
                citations = []
                answer_source = "system_fallback"
            elif generated and generated.evidences:
                message = _MSG_HAS_RESULTS
                citations = _build_citations_from_evidences(generated.evidences)
                answer_source = "evidence_fallback"
            else:
                message = _category_no_data_message(category_value)
                citations = []
                answer_source = "deterministic"

        elif generated and generated.evidences:
            # Case C: no LLM text but have document evidences
            message = _MSG_HAS_RESULTS
            citations = _build_citations_from_evidences(generated.evidences)
            answer_source = "evidence_fallback"

        else:
            # Case D: total fallback — no LLM, no evidences, no system
            citations = _build_citations(results)
            if citations:
                message = _MSG_HAS_RESULTS
                answer_source = "deterministic"
            elif allow_system_only:
                message = _category_no_data_message(category_value)
                answer_source = "deterministic"
            else:
                message = _MSG_NO_RESULTS
                answer_source = "deterministic"

        # ── Build AnswerSourceTrace (Phase 2C) ──────────────────────────
        _fallback = {
            "llm": 0, "evidence_fallback": 1,
            "system_fallback": 2, "deterministic": 3,
        }
        trace = AnswerSourceTrace(
            category=category_value,
            orchestration_ok=orchestration_ok,
            should_use_knowledge=should_use_knowledge,
            should_use_system_context=should_use_system_context,
            should_use_access_context=should_use_access_context,
            retrieval_skipped=retrieval_skipped,
            used_document_evidence=bool(generated and generated.evidences),
            used_system_context=bool(system_context_block),
            snippets_count=len(snippets),
            evidences_count=len(generated.evidences) if generated else 0,
            system_context_chars=len(system_context_block),
            answer_source=answer_source,
            fallback_level=_fallback.get(answer_source, 3),
            intent=generated.intent if generated else "none",
            llm_provider=generated.provider if generated else None,
            llm_model=generated.model if generated else None,
            used_history=generated.used_history if generated else False,
        )

        logger.info(
            "assistant.respond tenant_id=%s user_id=%d citations=%d %s",
            tenant_id,
            user_id,
            len(citations),
            " ".join(f"{k}={v}" for k, v in trace.to_log_dict().items()),
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
