"""
CTĐT Analysis Service — Orchestrate analysis tasks for an Update Cycle.

R4 Skeleton + R5 Draft mode.

Modes:
  - skeleton: All 7 tasks return status=needs_generation (no LLM).
  - draft: runs EvidenceAnalysisSkill for evidence_summary,
           CurrentCurriculumReviewSkill for evaluation_points, and
           ChangeProposalSkill for change_proposals.

Architecture::

    analyze_update_cycle()
        ├─ validate ai_document_ids scope (if provided)
        ├─ for each ANALYSIS_TASK:
        │    ├─ ctdt_retrieve(query_seed, task_type, ...)
        │    ├─ [draft + skilled task] → R5 skill adapter
        │    └─ [else] → collect contexts → AnalysisSkeletonItem
        └─ assemble AnalysisCycleResult with all 7 payload keys

Read-only against the DB. LLM calls only in draft mode with SYNTHESIS_ENABLED.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ctdt_retrieval_service import (
    CTDTRetrievalContext,
    CTDTRetrievalResult,
    CTDTTaskType,
    _get_scoped_document_ids,
    ctdt_retrieve,
)

logger = logging.getLogger(__name__)


# ── Query seeds — domain-specific prompts per task ───────────────────


ANALYSIS_TASK_SEEDS: dict[CTDTTaskType, str] = {
    CTDTTaskType.EVIDENCE_ANALYSIS: (
        "Thông tin, minh chứng, khảo sát, căn cứ thể hiện sự cần thiết "
        "phải cập nhật cải tiến chương trình đào tạo"
    ),
    CTDTTaskType.CURRENT_CURRICULUM_REVIEW: (
        "Đánh giá chương trình đào tạo hiện hành, mục tiêu, chuẩn đầu ra, "
        "nội dung chương trình, học phần và ma trận"
    ),
    CTDTTaskType.CHANGE_PROPOSAL: (
        "Nội dung cần cập nhật, bổ sung, thay đổi, cải tiến chất lượng "
        "chương trình đào tạo và phương án xử lý"
    ),
    CTDTTaskType.OBJECTIVE_SUGGESTION: (
        "Mục tiêu đào tạo, mục tiêu chung, mục tiêu cụ thể M1 M2 M3"
    ),
    CTDTTaskType.OUTCOME_SUGGESTION: (
        "Chuẩn đầu ra chương trình đào tạo C1 C2 C3 kiến thức kỹ năng "
        "năng lực tự chủ trách nhiệm"
    ),
    CTDTTaskType.COURSE_STRUCTURE: (
        "Nội dung chương trình, danh sách học phần, mã học phần, tín chỉ, "
        "lý thuyết, thực hành, học kỳ, tiên quyết"
    ),
    CTDTTaskType.MATRIX_MAPPING: (
        "Ma trận mục tiêu chuẩn đầu ra, ma trận học phần chuẩn đầu ra, "
        "mức đóng góp 1 2 3"
    ),
}

# Maps task_type → key in result_payload
TASK_PAYLOAD_KEY: dict[CTDTTaskType, str] = {
    CTDTTaskType.EVIDENCE_ANALYSIS: "evidence_summary",
    CTDTTaskType.CURRENT_CURRICULUM_REVIEW: "evaluation_points",
    CTDTTaskType.CHANGE_PROPOSAL: "change_proposals",
    CTDTTaskType.OBJECTIVE_SUGGESTION: "objective_suggestions",
    CTDTTaskType.OUTCOME_SUGGESTION: "outcome_suggestions",
    CTDTTaskType.COURSE_STRUCTURE: "course_change_suggestions",
    CTDTTaskType.MATRIX_MAPPING: "matrix_suggestions",
}

# Ordered list of analysis tasks
ANALYSIS_TASKS: list[CTDTTaskType] = [
    CTDTTaskType.EVIDENCE_ANALYSIS,
    CTDTTaskType.CURRENT_CURRICULUM_REVIEW,
    CTDTTaskType.CHANGE_PROPOSAL,
    CTDTTaskType.OBJECTIVE_SUGGESTION,
    CTDTTaskType.OUTCOME_SUGGESTION,
    CTDTTaskType.COURSE_STRUCTURE,
    CTDTTaskType.MATRIX_MAPPING,
]

ALL_PAYLOAD_KEYS = list(TASK_PAYLOAD_KEY.values())


# ── Result DTOs ──────────────────────────────────────────────────────


@dataclass
class AnalysisSource:
    """A single source reference from retrieval."""
    ai_document_id: int
    external_file_id: str | None
    filename: str | None
    document_role: str | None
    chunk_id: int
    chunk_index: int
    score: float
    quote: str
    update_cycle_id: str | None
    program_code: str | None


@dataclass
class AnalysisSkeletonItem:
    """A result item for one analysis task, optionally carrying draft payload."""
    status: str  # "needs_generation"
    task_type: str
    sources: list[AnalysisSource]
    payload: dict[str, Any] | None = None


@dataclass
class AnalysisSourceSummary:
    """Summary of all sources used across all tasks."""
    contexts_count: int
    documents_used: list[int]
    tasks_executed: list[str]
    latency_ms: int


@dataclass
class AnalysisCycleResult:
    """Full result from analyze_update_cycle."""
    update_cycle_id: str
    program_code: str | None
    program_name: str | None
    analysis_mode: str
    result_payload: dict[str, list[AnalysisSkeletonItem]]
    source_summary: AnalysisSourceSummary


# ── Validation ───────────────────────────────────────────────────────


class AnalysisValidationError(Exception):
    """Raised when analysis request validation fails."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


async def _validate_document_scope(
    db: AsyncSession,
    *,
    tenant_id: str,
    ai_document_ids: list[int],
    update_cycle_id: str,
    program_code: str | None = None,
) -> None:
    """
    Validate that all ai_document_ids belong to the specified update_cycle_id.

    Raises AnalysisValidationError if any document is out of scope.
    """
    scoped = await _get_scoped_document_ids(
        db,
        tenant_id=tenant_id,
        update_cycle_id=update_cycle_id,
        program_code=program_code,
    )
    scoped_ids = set(scoped.keys())
    out_of_scope = set(ai_document_ids) - scoped_ids

    if out_of_scope:
        raise AnalysisValidationError(
            code="invalid_document_scope",
            message=(
                f"Documents {sorted(out_of_scope)} do not belong to "
                f"update_cycle_id={update_cycle_id}"
                + (f" program_code={program_code}" if program_code else "")
            ),
        )


# ── Context → Source conversion ──────────────────────────────────────


def _context_to_source(ctx: CTDTRetrievalContext) -> AnalysisSource:
    """Convert a retrieval context to an AnalysisSource."""
    return AnalysisSource(
        ai_document_id=ctx.ai_document_id,
        external_file_id=ctx.external_file_id,
        filename=ctx.filename,
        document_role=ctx.document_role,
        chunk_id=ctx.chunk_id,
        chunk_index=ctx.chunk_index,
        score=ctx.score,
        quote=ctx.text[:500] if ctx.text else "",
        update_cycle_id=ctx.source.get("update_cycle_id") if isinstance(ctx.source, dict) else None,
        program_code=ctx.source.get("program_code") if isinstance(ctx.source, dict) else None,
    )


# ── Evidence skill adapter ───────────────────────────────────────────


async def _run_evidence_skill(
    *,
    update_cycle_id: str,
    program_code: str | None,
    program_name: str | None,
    sources: list[AnalysisSource],
) -> list[AnalysisSkeletonItem]:
    """
    Run EvidenceAnalysisSkill and convert result to AnalysisSkeletonItem list.

    Fail-open: if skill raises, returns empty skeleton with failed status.
    """
    from app.services.ctdt_skills.evidence_analysis_skill import (
        EvidenceAnalysisSkill,
        EvidenceAnalysisResult,
    )

    skill = EvidenceAnalysisSkill()

    try:
        result: EvidenceAnalysisResult = await skill.run(
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            program_name=program_name,
            sources=sources,
        )
    except Exception:
        logger.exception(
            "analysis.evidence_skill_error update_cycle=%s",
            update_cycle_id,
        )
        return [AnalysisSkeletonItem(
            status="failed",
            task_type="evidence_analysis",
            sources=sources,
        )]

    # If skill returned generated items, convert them
    if result.items:
        items = []
        for ev_item in result.items:
            items.append(AnalysisSkeletonItem(
                status=result.status,
                task_type="evidence_analysis",
                sources=ev_item.sources if ev_item.sources else sources,
            ))
        return items

    # No items -> single skeleton with skill status.
    return [AnalysisSkeletonItem(
        status=result.status,
        task_type="evidence_analysis",
        sources=sources,
    )]


# ── Curriculum review skill adapter ──────────────────────────────────────


async def _run_curriculum_review_skill(
    *,
    update_cycle_id: str,
    program_code: str | None,
    program_name: str | None,
    sources: list[AnalysisSource],
) -> list[AnalysisSkeletonItem]:
    """
    Run CurrentCurriculumReviewSkill and convert result to AnalysisSkeletonItem list.

    Fail-open: if skill raises, returns empty skeleton with failed status.
    """
    from app.services.ctdt_skills.current_curriculum_review_skill import (
        CurrentCurriculumReviewSkill,
        CurrentCurriculumReviewResult,
    )

    skill = CurrentCurriculumReviewSkill()

    try:
        result: CurrentCurriculumReviewResult = await skill.run(
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            program_name=program_name,
            sources=sources,
        )
    except Exception:
        logger.exception(
            "analysis.curriculum_review_skill_error update_cycle=%s",
            update_cycle_id,
        )
        return [AnalysisSkeletonItem(
            status="failed",
            task_type="current_curriculum_review",
            sources=sources,
        )]

    # If skill returned generated items, convert them
    if result.items:
        items = []
        for ev_item in result.items:
            items.append(AnalysisSkeletonItem(
                status=result.status,
                task_type="current_curriculum_review",
                sources=ev_item.sources if ev_item.sources else sources,
            ))
        return items

    # No items -> single skeleton with skill status.
    return [AnalysisSkeletonItem(
        status=result.status,
        task_type="current_curriculum_review",
        sources=sources,
    )]


# ── Change proposal skill adapter ────────────────────────────────────


async def _run_change_proposal_skill(
    *,
    update_cycle_id: str,
    program_code: str | None,
    program_name: str | None,
    sources: list[AnalysisSource],
) -> list[AnalysisSkeletonItem]:
    """
    Run ChangeProposalSkill and keep generated Mau 06 draft content in payload.

    Fail-open: if skill raises, returns empty skeleton with failed status.
    """
    from app.services.ctdt_skills.change_proposal_skill import (
        ChangeProposalSkill,
        ChangeProposalResult,
    )

    skill = ChangeProposalSkill()

    try:
        result: ChangeProposalResult = await skill.run(
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            program_name=program_name,
            sources=sources,
        )
    except Exception:
        logger.exception(
            "analysis.change_proposal_skill_error update_cycle=%s",
            update_cycle_id,
        )
        return [AnalysisSkeletonItem(
            status="failed",
            task_type="change_proposal",
            sources=sources,
            payload=None,
        )]

    # If skill returned generated items, preserve the draft fields for API clients.
    if result.items:
        items = []
        for proposal_item in result.items:
            items.append(AnalysisSkeletonItem(
                status=result.status,
                task_type="change_proposal",
                sources=proposal_item.sources if proposal_item.sources else sources,
                payload={
                    "target_area": proposal_item.target_area,
                    "change_type": proposal_item.change_type,
                    "current_issue": proposal_item.current_issue,
                    "proposed_change": proposal_item.proposed_change,
                    "rationale": proposal_item.rationale,
                    "expected_impact": proposal_item.expected_impact,
                    "priority": proposal_item.priority,
                    "confidence": proposal_item.confidence,
                },
            ))
        return items

    # No items -> single skeleton with skill status and no draft payload.
    return [AnalysisSkeletonItem(
        status=result.status,
        task_type="change_proposal",
        sources=sources,
        payload=None,
    )]


# ── Main orchestrator ────────────────────────────────────────────────


async def analyze_update_cycle(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int,
    update_cycle_id: str,
    program_id: str | None = None,
    program_code: str | None = None,
    program_name: str | None = None,
    ai_document_ids: list[int] | None = None,
    document_roles: list[str] | None = None,
    analysis_mode: str = "skeleton",
    top_k_per_task: int = 6,
    query_svc: Any = None,
) -> AnalysisCycleResult:
    """
    Orchestrate analysis of an update cycle across 7 CTĐT tasks.

    For each task:
      1. Use the task's query seed for retrieval.
      2. Call ctdt_retrieve() with scoped metadata.
      3. Collect sources into a skeleton item (status=needs_generation).

    If ai_document_ids is provided, validates scope before running tasks.

    Returns AnalysisCycleResult with all 7 payload keys always present.
    """
    t0 = time.monotonic()

    # ── Validate document scope if explicit IDs provided ─────────
    if ai_document_ids:
        await _validate_document_scope(
            db,
            tenant_id=tenant_id,
            ai_document_ids=ai_document_ids,
            update_cycle_id=update_cycle_id,
            program_code=program_code,
        )

    # ── Run all 7 tasks ──────────────────────────────────────────
    result_payload: dict[str, list[AnalysisSkeletonItem]] = {}
    all_doc_ids: set[int] = set()
    total_contexts = 0
    tasks_executed: list[str] = []

    for task_type in ANALYSIS_TASKS:
        query_seed = ANALYSIS_TASK_SEEDS[task_type]
        payload_key = TASK_PAYLOAD_KEY[task_type]

        logger.info(
            "analysis.task_start update_cycle=%s task=%s",
            update_cycle_id, task_type.value,
        )

        try:
            retrieval_result = await ctdt_retrieve(
                db,
                tenant_id=tenant_id,
                user_id=user_id,
                query=query_seed,
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_id=program_id,
                task_type=task_type,
                document_roles=document_roles,
                top_k=top_k_per_task,
                query_svc=query_svc,
            )
        except Exception:
            logger.exception(
                "analysis.task_failed update_cycle=%s task=%s",
                update_cycle_id, task_type.value,
            )
            # On failure, produce empty skeleton — don't abort entire analysis
            retrieval_result = CTDTRetrievalResult(
                query=query_seed,
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                task_type=task_type.value,
                document_roles_used=[],
                contexts=[],
                scoped_document_count=0,
                latency_ms=0,
            )

        # Convert contexts to sources
        sources = [_context_to_source(ctx) for ctx in retrieval_result.contexts]

        # If ai_document_ids was specified, filter sources to only those docs
        if ai_document_ids:
            allowed = set(ai_document_ids)
            sources = [s for s in sources if s.ai_document_id in allowed]

        # ── Draft mode: run skills for supported tasks ─────────────────
        if analysis_mode == "draft" and task_type == CTDTTaskType.EVIDENCE_ANALYSIS:
            skill_result = await _run_evidence_skill(
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_name=program_name,
                sources=sources,
            )
            result_payload[payload_key] = skill_result
        elif analysis_mode == "draft" and task_type == CTDTTaskType.CURRENT_CURRICULUM_REVIEW:
            skill_result = await _run_curriculum_review_skill(
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_name=program_name,
                sources=sources,
            )
            result_payload[payload_key] = skill_result
        elif analysis_mode == "draft" and task_type == CTDTTaskType.CHANGE_PROPOSAL:
            skill_result = await _run_change_proposal_skill(
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_name=program_name,
                sources=sources,
            )
            result_payload[payload_key] = skill_result
        else:
            # Build skeleton item
            skeleton_item = AnalysisSkeletonItem(
                status="needs_generation",
                task_type=task_type.value,
                sources=sources,
            )
            result_payload[payload_key] = [skeleton_item]

        # Track stats
        for s in sources:
            all_doc_ids.add(s.ai_document_id)
        total_contexts += len(sources)
        tasks_executed.append(task_type.value)

        logger.info(
            "analysis.task_done update_cycle=%s task=%s sources=%d",
            update_cycle_id, task_type.value, len(sources),
        )

    # ── Ensure all 7 payload keys exist (even if empty) ──────────
    for key in ALL_PAYLOAD_KEYS:
        if key not in result_payload:
            result_payload[key] = []

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    logger.info(
        "analysis.done update_cycle=%s program=%s tasks=%d "
        "total_contexts=%d documents_used=%d elapsed_ms=%d",
        update_cycle_id, program_code, len(tasks_executed),
        total_contexts, len(all_doc_ids), elapsed_ms,
    )

    return AnalysisCycleResult(
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        program_name=program_name,
        analysis_mode=analysis_mode,
        result_payload=result_payload,
        source_summary=AnalysisSourceSummary(
            contexts_count=total_contexts,
            documents_used=sorted(all_doc_ids),
            tasks_executed=tasks_executed,
            latency_ms=elapsed_ms,
        ),
    )
