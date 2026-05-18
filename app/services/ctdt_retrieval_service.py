"""
CTDT Retrieval Service — metadata-aware retrieval for CTĐT documents.

R3: Scoped retrieval that enforces update_cycle_id isolation,
document_role filtering, and program_code narrowing.

Architecture:
    1. Pre-filter: query Document table by CTĐT metadata
       (update_cycle_id, program_code, document_role) to get allowed doc IDs.
    2. Delegate: pass restricted doc IDs into the existing QueryService pipeline.
    3. Enrich: attach CTĐT source metadata to each result context.

Guards:
    - Never returns documents from a different update_cycle.
    - Never falls back to global search if scoped results are empty.
    - Empty contexts = no matching documents (explicit, not silent).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.services.retrieval.types import QueryResult

logger = logging.getLogger(__name__)


# ── CTĐT Task Types ──────────────────────────────────────────────────


class CTDTTaskType(str, Enum):
    """Task types that determine which document_roles to prioritize."""
    GENERAL_QUERY = "general_query"
    EVIDENCE_ANALYSIS = "evidence_analysis"
    CURRENT_CURRICULUM_REVIEW = "current_curriculum_review"
    CHANGE_PROPOSAL = "change_proposal"
    OBJECTIVE_SUGGESTION = "objective_suggestion"
    OUTCOME_SUGGESTION = "outcome_suggestion"
    COURSE_STRUCTURE = "course_structure"
    MATRIX_MAPPING = "matrix_mapping"
    TEMPLATE_LOOKUP = "template_lookup"
    CURRICULUM_UPDATE_DESIGN = "curriculum_update_design"


# ── Document Role Policy ─────────────────────────────────────────────

# Maps task_type → ordered list of preferred document_roles.
# When request has explicit document_roles, those override the policy.

TASK_ROLE_POLICY: dict[CTDTTaskType, list[str]] = {
    CTDTTaskType.GENERAL_QUERY: [],  # empty = all roles in scope

    CTDTTaskType.EVIDENCE_ANALYSIS: [
        "survey_evidence",
        "meeting_report",
        "direction_decision",
        "legal_regulation",
    ],

    CTDTTaskType.CURRENT_CURRICULUM_REVIEW: [
        "current_curriculum",
        "comparison_report",
        "legal_regulation",
    ],

    CTDTTaskType.CHANGE_PROPOSAL: [
        "current_curriculum",
        "comparison_report",
        "survey_evidence",
        "meeting_report",
        "legal_regulation",
    ],

    CTDTTaskType.OBJECTIVE_SUGGESTION: [
        "current_curriculum",
        "survey_evidence",
        "comparison_report",
        "legal_regulation",
    ],

    CTDTTaskType.OUTCOME_SUGGESTION: [
        "current_curriculum",
        "course_syllabus",
        "legal_regulation",
        "survey_evidence",
    ],

    CTDTTaskType.COURSE_STRUCTURE: [
        "current_curriculum",
        "course_syllabus",
        "comparison_report",
    ],

    CTDTTaskType.MATRIX_MAPPING: [
        "current_curriculum",
        "course_syllabus",
    ],

    CTDTTaskType.TEMPLATE_LOOKUP: [
        "template",
        "legal_regulation",
    ],

    CTDTTaskType.CURRICULUM_UPDATE_DESIGN: [
        "direction_decision",
        "legal_regulation",
        "current_curriculum",
        "survey_evidence",
        "meeting_report",
        "comparison_report",
        "course_syllabus",
        "other",
    ],
}


def resolve_document_roles(
    task_type: CTDTTaskType,
    explicit_roles: list[str] | None,
) -> list[str]:
    """
    Resolve effective document_roles for retrieval.

    Rules:
      - If explicit_roles is provided and non-empty → use those (override).
      - Otherwise → use policy for task_type.
      - Empty list = no role filter (all roles in scope).
    """
    if explicit_roles:
        return explicit_roles
    return TASK_ROLE_POLICY.get(task_type, [])


# ── Result DTO ────────────────────────────────────────────────────────


@dataclass
class CTDTRetrievalContext:
    """A single retrieval context with CTĐT source metadata."""
    ai_document_id: int
    external_file_id: str | None
    filename: str | None
    document_role: str | None
    chunk_id: int
    chunk_index: int
    score: float
    text: str
    source: dict[str, Any]


@dataclass
class CTDTRetrievalResult:
    """Full result from CTĐT scoped retrieval."""
    query: str
    update_cycle_id: str
    program_code: str | None
    task_type: str
    document_roles_used: list[str]
    contexts: list[CTDTRetrievalContext]
    scoped_document_count: int
    latency_ms: int


# ── Pre-filter: scope documents by CTĐT metadata ─────────────────────


async def _get_scoped_document_ids(
    db: AsyncSession,
    *,
    tenant_id: str,
    update_cycle_id: str,
    program_code: str | None = None,
    program_id: str | None = None,
    document_roles: list[str] | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Query Document table for CTĐT documents matching the scope.

    Returns dict mapping doc_id → ctdt metadata dict.
    Only returns documents with status in ('ready', 'indexed').

    Filters:
      - source = 'cn_ctdt'
      - meta->ctdt->update_cycle_id matches
      - meta->ctdt->program_code matches (if provided)
      - meta->ctdt->program_id matches (if provided)
      - meta->ctdt->document_role in document_roles (if provided)
    """
    # Base query: tenant + CTĐT source + queryable status
    stmt = (
        select(Document)
        .where(
            Document.tenant_id == tenant_id,
            Document.source == "cn_ctdt",
            Document.status.in_(("ready", "indexed")),
        )
    )

    result = await db.execute(stmt)
    docs = result.scalars().all()

    scoped: dict[int, dict[str, Any]] = {}

    for doc in docs:
        meta = doc.meta or {}
        ctdt = meta.get("ctdt", {})

        # Hard filter: update_cycle_id must match
        if str(ctdt.get("update_cycle_id", "")) != str(update_cycle_id):
            continue

        # Optional filter: program_code
        if program_code:
            doc_program_code = ctdt.get("program_code")
            if doc_program_code and str(doc_program_code) != str(program_code):
                continue

        # Optional filter: program_id
        if program_id:
            doc_program_id = ctdt.get("program_id")
            if doc_program_id and str(doc_program_id) != str(program_id):
                continue

        # Optional filter: document_role
        if document_roles:
            doc_role = ctdt.get("document_role", "")
            if doc_role not in document_roles:
                continue

        scoped[doc.id] = {
            "external_file_id": ctdt.get("external_file_id"),
            "filename": doc.title,
            "document_role": ctdt.get("document_role"),
            "update_cycle_id": ctdt.get("update_cycle_id"),
            "program_code": ctdt.get("program_code"),
            "program_id": ctdt.get("program_id"),
            "program_name": ctdt.get("program_name"),
        }

    return scoped


# ── Enrich results with CTĐT metadata ────────────────────────────────


def _enrich_results(
    results: list[QueryResult],
    doc_metadata: dict[int, dict[str, Any]],
) -> list[CTDTRetrievalContext]:
    """
    Convert QueryResult list to CTDTRetrievalContext list,
    attaching CTĐT source metadata.

    Only includes results whose document_id is in doc_metadata
    (defense-in-depth against scope leakage).
    """
    contexts: list[CTDTRetrievalContext] = []

    for r in results:
        meta = doc_metadata.get(r.document_id)
        if meta is None:
            # Document not in CTĐT scope — skip (guard against leakage)
            logger.warning(
                "ctdt_retrieval.scope_leak_prevented doc_id=%d",
                r.document_id,
            )
            continue

        # Extract chunk_index from synthetic chunk_id
        chunk_index = r.chunk_id % 100_000 if r.chunk_id else 0

        contexts.append(CTDTRetrievalContext(
            ai_document_id=r.document_id,
            external_file_id=meta.get("external_file_id"),
            filename=meta.get("filename"),
            document_role=meta.get("document_role"),
            chunk_id=r.chunk_id,
            chunk_index=chunk_index,
            score=round(r.score, 4),
            text=r.snippet or "",
            source={
                "update_cycle_id": meta.get("update_cycle_id"),
                "program_code": meta.get("program_code"),
                "program_id": meta.get("program_id"),
                "section": None,
                "page": None,
            },
        ))

    return contexts


# ── Main retrieve function ────────────────────────────────────────────


async def ctdt_retrieve(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int,
    query: str,
    update_cycle_id: str,
    program_code: str | None = None,
    program_id: str | None = None,
    task_type: CTDTTaskType = CTDTTaskType.GENERAL_QUERY,
    document_roles: list[str] | None = None,
    top_k: int = 8,
    query_svc: Any = None,
) -> CTDTRetrievalResult:
    """
    CTĐT-scoped retrieval: pre-filter by metadata, then delegate to QueryService.

    Steps:
      1. Resolve effective document_roles from task_type + explicit roles.
      2. Pre-filter: query Document table for matching CTĐT docs.
      3. If no docs in scope → return empty (never fallback).
      4. Delegate to QueryService with restricted doc IDs.
      5. Enrich results with CTĐT source metadata.
      6. Defense-in-depth: strip any result outside scope.
    """
    t0 = time.monotonic()

    # Step 1: Resolve roles
    effective_roles = resolve_document_roles(task_type, document_roles)

    # Step 2: Pre-filter
    doc_metadata = await _get_scoped_document_ids(
        db,
        tenant_id=tenant_id,
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        program_id=program_id,
        document_roles=effective_roles if effective_roles else None,
    )

    scoped_doc_ids = set(doc_metadata.keys())

    logger.info(
        "ctdt_retrieval.scoped tenant_id=%s update_cycle=%s "
        "program_code=%s task_type=%s roles=%s scoped_docs=%d",
        tenant_id, update_cycle_id, program_code,
        task_type.value, effective_roles or "all",
        len(scoped_doc_ids),
    )

    # Step 3: No docs → empty result (never fallback)
    if not scoped_doc_ids:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        return CTDTRetrievalResult(
            query=query,
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            task_type=task_type.value,
            document_roles_used=effective_roles,
            contexts=[],
            scoped_document_count=0,
            latency_ms=elapsed_ms,
        )

    # Step 4: Delegate to QueryService with scoped doc IDs
    # R3.1: allowed_document_ids restricts retrieval at the vector/BM25
    # layer, not just post-filter. This prevents recall loss when
    # top_k raw results would otherwise be dominated by out-of-scope docs.
    if query_svc is None:
        from app.services.retrieval.factories import get_query_service
        query_svc = get_query_service()

    results = await query_svc.query(
        tenant_id=tenant_id,
        user_id=user_id,
        query_text=query,
        final_limit=top_k,
        mode="hybrid",
        include_debug=False,
        history=[],
        allowed_document_ids=scoped_doc_ids,
    )

    # Step 5 + 6: Enrich + defense-in-depth scope filter
    contexts = _enrich_results(results, doc_metadata)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    logger.info(
        "ctdt_retrieval.done tenant_id=%s update_cycle=%s "
        "scoped=%d raw_results=%d filtered_contexts=%d elapsed_ms=%d",
        tenant_id, update_cycle_id,
        len(scoped_doc_ids), len(results), len(contexts),
        elapsed_ms,
    )

    return CTDTRetrievalResult(
        query=query,
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        task_type=task_type.value,
        document_roles_used=effective_roles,
        contexts=contexts,
        scoped_document_count=len(scoped_doc_ids),
        latency_ms=elapsed_ms,
    )
