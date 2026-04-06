"""
Phase 1.2A — Eval runner (orchestration).

Loads dataset, executes cases through the appropriate flow
(assistant or query), grades results, and produces output.

This module is used by the CLI script but can also be called
programmatically from tests or other automation.

Design rules:
  - Grading is always done with programmatic graders (no LLM).
  - Retrieval and answer metrics are computed separately.
  - Each case is graded immediately after execution.
  - Runner never mutates the existing query/assistant flows.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.evals.dataset import DatasetLoadResult, load_dataset
from app.evals.graders import programmatic as graders
from app.evals.schemas import (
    AnswerMetrics,
    CaseResult,
    EvalCase,
    RetrievalMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for a single eval run."""

    dataset_path: str = "evals/phase1/golden_set.jsonl"
    mode: str = "assistant"  # "assistant" | "query"
    tenant: str | None = None
    case_id: str | None = None
    slice_tag: str | None = None
    limit: int | None = None
    output_dir: str = "evals/outputs/phase1"

    # Execution mode
    exec_mode: str = "api"  # "api" | "direct"
    base_url: str = "http://localhost:8000"
    api_key: str = ""


@dataclass
class RunResult:
    """Output of a complete eval run."""

    config: RunConfig
    dataset: DatasetLoadResult = field(default_factory=DatasetLoadResult)
    results: list[CaseResult] = field(default_factory=list)
    elapsed_ms: int = 0


async def run_eval(config: RunConfig) -> RunResult:
    """Execute a full eval run.

    Steps:
      1. Load and validate dataset
      2. Filter cases by config
      3. Execute each case
      4. Grade each case
      5. Return results
    """
    run_result = RunResult(config=config)
    t0 = time.perf_counter()

    # ── Load dataset ──────────────────────────────────────────────────
    dataset = load_dataset(
        config.dataset_path,
        tenant_id=config.tenant,
        endpoint_mode=config.mode if config.mode != "mixed" else None,
        case_id=config.case_id,
        slice_tag=config.slice_tag,
        limit=config.limit,
    )
    run_result.dataset = dataset

    if dataset.errors:
        for err in dataset.errors:
            logger.warning("Dataset error at line %s: %s", err.get("line"), err.get("error"))

    logger.info(
        "Loaded %d cases (valid=%d, errors=%d, filtered_out=%d)",
        len(dataset.cases), dataset.valid_lines,
        len(dataset.errors), dataset.skipped_by_filter,
    )

    if not dataset.cases:
        logger.warning("No cases to evaluate after filtering")
        run_result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return run_result

    # ── Execute cases ─────────────────────────────────────────────────
    for i, case in enumerate(dataset.cases, start=1):
        logger.info(
            "[%d/%d] case=%s tenant=%s mode=%s type=%s",
            i, len(dataset.cases),
            case.case_id, case.tenant_id,
            case.endpoint_mode, case.expected_answer_type,
        )

        case_result = await _execute_and_grade(case, config)
        run_result.results.append(case_result)

        # Log per-case summary
        _log_case_result(case_result)

    run_result.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return run_result


async def _execute_and_grade(case: EvalCase, config: RunConfig) -> CaseResult:
    """Execute a single case and grade the output."""
    result = CaseResult(
        case_id=case.case_id,
        tenant_id=case.tenant_id,
        mode=case.endpoint_mode,
        question=case.question,
        expected_answer_type=case.expected_answer_type,
        expected_source_document_ids=case.expected_source_document_ids,
        slice_tags=case.slice_tags,
        criticality=case.criticality,
    )

    t0 = time.perf_counter()

    try:
        if config.exec_mode == "api":
            raw_output = await _execute_api(case, config)
        else:
            raw_output = await _execute_direct(case)

        result.latency_ms = int((time.perf_counter() - t0) * 1000)

        # Extract raw outputs
        result.answer_text = raw_output.get("answer_text")
        result.citation_document_ids = raw_output.get("citation_document_ids", [])
        result.retrieved_document_ids = raw_output.get("retrieved_document_ids", [])

    except Exception as exc:
        result.latency_ms = int((time.perf_counter() - t0) * 1000)
        result.error = f"{exc.__class__.__name__}: {exc}"
        result.failure_reasons.append(f"Execution error: {exc}")
        return result

    # ── Grade ─────────────────────────────────────────────────────────
    _grade_retrieval(case, result)
    _grade_answer(case, result, raw_output)

    return result


def _grade_retrieval(case: EvalCase, result: CaseResult) -> None:
    """Apply all retrieval graders to a case result."""
    doc_ids = result.retrieved_document_ids or result.citation_document_ids
    expected = case.expected_source_document_ids

    result.retrieval = RetrievalMetrics(
        retrieved_document_ids=doc_ids,
        hit_at_1=graders.hit_at_k(doc_ids, expected, 1),
        hit_at_3=graders.hit_at_k(doc_ids, expected, 3),
        hit_at_5=graders.hit_at_k(doc_ids, expected, 5),
        recall_at_5=graders.recall_at_k(doc_ids, expected, 5),
        mrr=graders.mrr(doc_ids, expected),
    )


def _grade_answer(case: EvalCase, result: CaseResult, raw_output: dict) -> None:
    """Apply all answer graders to a case result."""
    answer_text = result.answer_text
    answer_result = raw_output  # The full response dict for citation checks

    # Keyword coverage
    kw_coverage, kw_found, kw_missing = graders.keyword_coverage(
        answer_text, case.expected_keywords,
    )

    # Forbidden keywords
    forbidden_viol, forbidden_found = graders.forbidden_keyword_violation(
        answer_text, case.forbidden_keywords,
    )

    result.answer = AnswerMetrics(
        has_answer=graders.has_answer(answer_text),
        has_citations=graders.has_citations(answer_result),
        citation_doc_ids_valid=graders.citation_doc_ids_exist(answer_result),
        citation_same_tenant=graders.citations_same_tenant(
            answer_result, case.tenant_id,
        ),
        keyword_coverage=round(kw_coverage, 4),
        keywords_found=kw_found,
        keywords_missing=kw_missing,
        forbidden_keyword_violation=forbidden_viol,
        forbidden_keywords_found=forbidden_found,
        abstention_detected=graders.abstention_detected(answer_text),
        abstention_behavior_correct=graders.abstention_behavior_basic(
            answer_text, case.expected_answer_type,
        ),
    )


# ── Execution backends ────────────────────────────────────────────────────────


async def _execute_api(case: EvalCase, config: RunConfig) -> dict:
    """Execute a case via the running API server."""
    import httpx

    if case.endpoint_mode == "query":
        url = f"{config.base_url}/api/v1/query"
        payload = {
            "query": case.question,
            "history": case.history,
            "include_debug": True,
        }
    else:
        url = f"{config.base_url}/api/v1/assistant/respond"
        payload = {
            "message": case.question,
            "history": case.history,
        }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    return _normalize_api_response(data, case.endpoint_mode)


async def _execute_direct(case: EvalCase) -> dict:
    """Execute a case in-process via service calls."""
    if case.endpoint_mode == "query":
        return await _execute_direct_query(case)
    else:
        return await _execute_direct_assistant(case)


async def _execute_direct_assistant(case: EvalCase) -> dict:
    """Execute assistant mode in-process."""
    from app.schemas.assistant import AssistantRespondRequest
    from app.services.assistant_service import AssistantService

    svc = AssistantService()
    request = AssistantRespondRequest(
        message=case.question,
        history=case.history,
    )

    response = await svc.respond(
        request=request,
        tenant_id=case.tenant_id,
        user_id=0,
        trace_id=f"eval-{case.case_id}",
    )

    citations = [
        {
            "document_id": c.document_id,
            "source_document_id": c.source_document_id,
            "chunk_id": c.chunk_id,
            "snippet": c.snippet,
            "score": c.score,
        }
        for c in response.citations
    ]

    return {
        "answer_text": response.message,
        "citations": citations,
        "citation_document_ids": [
            c.source_document_id or c.document_id
            for c in response.citations
            if c.document_id or c.source_document_id
        ],
        "retrieved_document_ids": [
            c.source_document_id or c.document_id
            for c in response.citations
            if c.document_id or c.source_document_id
        ],
    }


async def _execute_direct_query(case: EvalCase) -> dict:
    """Execute query mode in-process."""
    from app.services.retrieval.factories import get_query_service

    query_svc = get_query_service()
    results = await query_svc.query(
        tenant_id=case.tenant_id,
        user_id=0,
        query_text=case.question,
        mode="hybrid",
        include_debug=True,
    )

    return {
        "answer_text": None,  # Query mode doesn't produce answers
        "citations": [],
        "citation_document_ids": [],
        "retrieved_document_ids": [
            getattr(r, "source_document_id", None) or r.document_id
            for r in results
        ],
    }


def _normalize_api_response(data: dict, endpoint_mode: str) -> dict:
    """Normalize API response to a consistent format."""
    if endpoint_mode == "query":
        results = data.get("results", [])
        return {
            "answer_text": data.get("answer"),
            "citations": [],
            "citation_document_ids": [],
            "retrieved_document_ids": [
                r.get("source_document_id") or r.get("document_id")
                for r in results
                if r.get("document_id")
            ],
        }
    else:
        # Assistant mode
        citations = data.get("citations", [])
        return {
            "answer_text": data.get("message"),
            "citations": citations,
            "citation_document_ids": [
                c.get("source_document_id") or c.get("document_id")
                for c in citations
                if c.get("document_id") or c.get("source_document_id")
            ],
            "retrieved_document_ids": [
                c.get("source_document_id") or c.get("document_id")
                for c in citations
                if c.get("document_id") or c.get("source_document_id")
            ],
        }


def _log_case_result(result: CaseResult) -> None:
    """Log a brief per-case summary."""
    status = "OK" if not result.error else f"ERR: {result.error[:60]}"
    logger.info(
        "  → %s | answer=%s cit=%s kw=%.2f hit@3=%s lat=%dms",
        status,
        "yes" if result.answer.has_answer else "no",
        "yes" if result.answer.has_citations else "no",
        result.answer.keyword_coverage,
        "yes" if result.retrieval.hit_at_3 else "no",
        result.latency_ms,
    )
