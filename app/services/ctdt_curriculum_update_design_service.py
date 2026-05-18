"""
Curriculum Update Design Service — Orchestrate R6.0 design draft generation.

Retrieves CTĐT documents → detects missing document types →
calls CurriculumUpdateDesignSkill → optionally saves draft.

Guards:
  - No writes to Program / ProgramVersion / ProgramVersionRevision.
  - Draft saved only to ctdt_analysis_drafts with draft_type="curriculum_update_design".
  - Fail-open: LLM errors → return payload with warnings, never crash.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ctdt_retrieval_service import (
    CTDTRetrievalContext,
    CTDTRetrievalResult,
    CTDTTaskType,
    ctdt_retrieve,
)

logger = logging.getLogger(__name__)


# ── Query seed for curriculum update design ──────────────────────────

_DESIGN_QUERY_SEED = (
    "Phân tích yêu cầu cập nhật CTĐT, CTĐT hiện hành, minh chứng đánh giá, "
    "đề xuất cải tiến mục tiêu, chuẩn đầu ra, cấu trúc, học phần, ma trận."
)

# Document roles that we check for and flag as missing_information
_CRITICAL_ROLES = {
    "current_curriculum": "Không tìm thấy tài liệu CTĐT hiện hành trong phạm vi đợt cập nhật.",
    "direction_decision": "Không tìm thấy quyết định/chỉ đạo cập nhật CTĐT.",
    "legal_regulation": "Không tìm thấy văn bản pháp lý/quy định liên quan.",
}

DRAFT_TYPE = "curriculum_update_design"


# ── Result DTO ───────────────────────────────────────────────────────


@dataclass
class DesignDraftSourceSummary:
    """Summary of sources used."""
    contexts_count: int
    documents_used: list[int]
    tasks_executed: list[str]
    latency_ms: int


@dataclass
class DesignDraftResult:
    """Full result from generate_curriculum_update_design_draft."""
    update_cycle_id: str
    program_code: str | None
    program_name: str | None
    draft_type: str
    draft_id: int | None
    draft_saved: bool
    payload: dict[str, Any]
    source_summary: DesignDraftSourceSummary


# ── Context → Source conversion ──────────────────────────────────────


def _context_to_source(ctx: CTDTRetrievalContext):
    """Convert retrieval context to AnalysisSource (lazy import to avoid circular)."""
    from app.services.ctdt_analysis_service import AnalysisSource

    return AnalysisSource(
        ai_document_id=ctx.ai_document_id,
        external_file_id=ctx.external_file_id,
        filename=ctx.filename,
        document_role=ctx.document_role,
        chunk_id=ctx.chunk_id,
        chunk_index=ctx.chunk_index,
        score=round(ctx.score, 4),
        quote=ctx.text[:500] if ctx.text else "",
        update_cycle_id=ctx.source.get("update_cycle_id") if isinstance(ctx.source, dict) else None,
        program_code=ctx.source.get("program_code") if isinstance(ctx.source, dict) else None,
    )


# ── Missing info detection ───────────────────────────────────────────


def _detect_missing_roles(
    contexts: list[CTDTRetrievalContext],
) -> list[dict[str, str]]:
    """Detect which critical document_roles are absent from contexts."""
    found_roles = {ctx.document_role for ctx in contexts if ctx.document_role}
    missing = []
    for role, description in _CRITICAL_ROLES.items():
        if role not in found_roles:
            missing.append({"type": role, "description": description})
    return missing


# ── Main orchestrator ────────────────────────────────────────────────


async def generate_curriculum_update_design_draft(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int,
    update_cycle_id: str,
    program_id: str | None = None,
    program_code: str | None = None,
    program_name: str | None = None,
    top_k: int = 12,
    user_instruction: str | None = None,
    save_draft: bool = False,
    query_svc: Any = None,
) -> DesignDraftResult:
    """
    Orchestrate curriculum update design draft generation.

    Steps:
      1. Retrieve contexts with CURRICULUM_UPDATE_DESIGN task type.
      2. Detect missing document roles.
      3. Call CurriculumUpdateDesignSkill.
      4. Merge detected missing info into skill output.
      5. Optionally save draft.
    """
    t0 = time.monotonic()

    # ── Step 1: Retrieve ─────────────────────────────────────────
    try:
        retrieval_result: CTDTRetrievalResult = await ctdt_retrieve(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            query=_DESIGN_QUERY_SEED,
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            program_id=program_id,
            task_type=CTDTTaskType.CURRICULUM_UPDATE_DESIGN,
            top_k=top_k,
            query_svc=query_svc,
        )
    except Exception:
        logger.exception(
            "design_draft.retrieval_failed update_cycle=%s",
            update_cycle_id,
        )
        retrieval_result = CTDTRetrievalResult(
            query=_DESIGN_QUERY_SEED,
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            task_type=CTDTTaskType.CURRICULUM_UPDATE_DESIGN.value,
            document_roles_used=[],
            contexts=[],
            scoped_document_count=0,
            latency_ms=0,
        )

    # ── Step 2: Detect missing roles ─────────────────────────────
    pre_missing = _detect_missing_roles(retrieval_result.contexts)

    # ── Step 3: Convert contexts → sources ───────────────────────
    sources = [_context_to_source(ctx) for ctx in retrieval_result.contexts]
    all_doc_ids = sorted({s.ai_document_id for s in sources})

    # ── Step 4: Call skill ───────────────────────────────────────
    from app.services.ctdt_skills.curriculum_update_design_skill import (
        CurriculumUpdateDesignSkill,
        CurriculumUpdateDesignResult,
    )

    skill = CurriculumUpdateDesignSkill()

    try:
        skill_result: CurriculumUpdateDesignResult = await skill.run(
            update_cycle_id=update_cycle_id,
            program_id=program_id,
            program_code=program_code,
            program_name=program_name,
            sources=sources,
            user_instruction=user_instruction,
        )
    except Exception:
        logger.exception(
            "design_draft.skill_failed update_cycle=%s",
            update_cycle_id,
        )
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignPayload,
            DesignStatus,
        )
        skill_result = CurriculumUpdateDesignResult(
            status=DesignStatus.FAILED.value,
            payload=CurriculumUpdateDesignPayload(
                missing_information=pre_missing,
            ),
            warnings=["Skill xử lý thất bại."],
        )

    # ── Step 5: Merge pre-detected missing info ──────────────────
    existing_types = {
        mi.get("type") for mi in skill_result.payload.missing_information
    }
    for mi in pre_missing:
        if mi["type"] not in existing_types:
            skill_result.payload.missing_information.append(mi)

    # ── Build payload dict ───────────────────────────────────────
    payload = asdict(skill_result.payload)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # ── Step 6: Optional draft save ──────────────────────────────
    draft_id = None
    draft_saved = False

    if save_draft:
        from app.db.models.ctdt_analysis_draft import CTDTAnalysisDraft

        try:
            draft = CTDTAnalysisDraft(
                tenant_id=tenant_id,
                update_cycle_id=update_cycle_id,
                program_id=program_id,
                program_code=program_code,
                program_name=program_name,
                analysis_mode="design",
                draft_type=DRAFT_TYPE,
                result_payload=payload,
                source_summary={
                    "contexts_count": len(sources),
                    "documents_used": all_doc_ids,
                    "tasks_executed": ["curriculum_update_design"],
                    "latency_ms": elapsed_ms,
                },
                created_by=user_id,
                updated_by=user_id,
                status="draft",
            )
            db.add(draft)
            await db.flush()
            await db.refresh(draft)
            await db.commit()
            draft_id = draft.id
            draft_saved = True

            logger.info(
                "design_draft.saved draft_id=%d update_cycle=%s",
                draft_id, update_cycle_id,
            )
        except Exception:
            logger.exception(
                "design_draft.save_failed update_cycle=%s",
                update_cycle_id,
            )
            try:
                await db.rollback()
            except Exception:
                logger.exception(
                    "design_draft.rollback_failed update_cycle=%s",
                    update_cycle_id,
                )
            raise

    logger.info(
        "design_draft.done update_cycle=%s program=%s "
        "contexts=%d documents=%d elapsed_ms=%d saved=%s",
        update_cycle_id, program_code,
        len(sources), len(all_doc_ids), elapsed_ms, draft_saved,
    )

    return DesignDraftResult(
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        program_name=program_name,
        draft_type=DRAFT_TYPE,
        draft_id=draft_id,
        draft_saved=draft_saved,
        payload=payload,
        source_summary=DesignDraftSourceSummary(
            contexts_count=len(sources),
            documents_used=all_doc_ids,
            tasks_executed=["curriculum_update_design"],
            latency_ms=elapsed_ms,
        ),
    )
