"""
Outcome Update Service — Orchestrate R6.2B outcome update draft generation.

Context pack (R6.2A) → OutcomeUpdateSkill → optionally save draft.

Guards:
  - No writes to Program / ProgramVersion / ProgramVersionRevision.
  - Draft saved only to ctdt_analysis_drafts with draft_type="outcome_update".
  - Fail-open: LLM errors → return payload with warnings, never crash.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ctdt_outcome_context_service import build_outcome_update_context_pack

logger = logging.getLogger(__name__)
DRAFT_TYPE = "outcome_update"


@dataclass
class OutcomeSourceSummary:
    contexts_count: int
    documents_used: list[int]
    tasks_executed: list[str]
    latency_ms: int


@dataclass
class OutcomeDraftResult:
    update_cycle_id: str
    program_code: str | None
    program_name: str | None
    draft_type: str
    draft_id: int | None
    draft_saved: bool
    payload: dict[str, Any]
    context_pack_summary: dict[str, Any]
    source_summary: OutcomeSourceSummary
    generation_status: str = "needs_generation"
    warnings: list[str] | None = None


async def generate_outcome_update_draft(
    db: AsyncSession, *, tenant_id: str, user_id: int, update_cycle_id: str,
    program_id: str | None = None, program_code: str | None = None,
    program_name: str | None = None, top_k_per_role: int = 5,
    user_instruction: str | None = None, save_draft: bool = False,
    query_svc: Any = None,
) -> OutcomeDraftResult:
    t0 = time.monotonic()

    context_pack = await build_outcome_update_context_pack(
        db, tenant_id=tenant_id, user_id=user_id,
        update_cycle_id=update_cycle_id, program_id=program_id,
        program_code=program_code, program_name=program_name,
        top_k_per_role=top_k_per_role, query_svc=query_svc,
    )

    total_contexts = (
        len(context_pack.current_outcome_contexts)
        + len(context_pack.current_curriculum_contexts)
        + len(context_pack.direction_contexts)
        + len(context_pack.legal_contexts)
        + len(context_pack.evidence_contexts)
        + len(context_pack.comparison_contexts)
        + len(context_pack.course_syllabus_contexts)
    )
    all_doc_ids = context_pack.source_summary.documents_used if context_pack.source_summary else []

    from app.services.ctdt_skills.outcome_update_skill import (
        OutcomeUpdateSkill, OutcomeUpdateResult, OutcomeUpdatePayload, OutcomeUpdateStatus,
    )

    skill = OutcomeUpdateSkill()
    try:
        skill_result: OutcomeUpdateResult = await skill.run(
            update_cycle_id=update_cycle_id, program_id=program_id,
            program_code=program_code, program_name=program_name,
            context_pack=context_pack, user_instruction=user_instruction,
        )
    except Exception:
        logger.exception("outcome_update.skill_failed update_cycle=%s", update_cycle_id)
        skill_result = OutcomeUpdateResult(
            status=OutcomeUpdateStatus.FAILED,
            payload=OutcomeUpdatePayload(missing_information=list(context_pack.missing_information)),
            warnings=["Skill xử lý thất bại."],
        )

    existing_types = {mi.get("type") for mi in skill_result.payload.missing_information}
    for mi in context_pack.missing_information:
        if mi["type"] not in existing_types:
            skill_result.payload.missing_information.append(mi)

    payload = asdict(skill_result.payload)
    generation_status = skill_result.status
    generation_warnings = list(skill_result.warnings)

    context_pack_summary: dict[str, Any] = {"role_coverage": {}, "missing_information": list(context_pack.missing_information)}
    for key, cov in context_pack.role_coverage.items():
        context_pack_summary["role_coverage"][key] = {
            "document_roles": cov.document_roles, "context_count": cov.context_count,
            "documents_used": cov.documents_used, "status": cov.status,
            "scoped_document_count": cov.scoped_document_count, "retrieval_status": cov.retrieval_status,
        }

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    draft_id = None
    draft_saved = False

    if save_draft:
        from app.db.models.ctdt_analysis_draft import CTDTAnalysisDraft
        try:
            draft = CTDTAnalysisDraft(
                tenant_id=tenant_id, update_cycle_id=update_cycle_id,
                program_id=program_id, program_code=program_code,
                program_name=program_name, analysis_mode="design",
                draft_type=DRAFT_TYPE,
                result_payload={**payload, "_meta": {"generation_status": generation_status, "warnings": generation_warnings}},
                source_summary={"contexts_count": total_contexts, "documents_used": all_doc_ids, "tasks_executed": ["outcome_update"], "latency_ms": elapsed_ms},
                created_by=user_id, updated_by=user_id, status="draft",
            )
            db.add(draft)
            await db.flush()
            await db.refresh(draft)
            await db.commit()
            draft_id = draft.id
            draft_saved = True
            logger.info("outcome_update.saved draft_id=%d update_cycle=%s", draft_id, update_cycle_id)
        except Exception:
            logger.exception("outcome_update.save_failed update_cycle=%s", update_cycle_id)
            try:
                await db.rollback()
            except Exception:
                logger.exception("outcome_update.rollback_failed update_cycle=%s", update_cycle_id)
            raise

    logger.info("outcome_update.done update_cycle=%s program=%s contexts=%d documents=%d elapsed_ms=%d saved=%s",
                update_cycle_id, program_code, total_contexts, len(all_doc_ids), elapsed_ms, draft_saved)

    return OutcomeDraftResult(
        update_cycle_id=update_cycle_id, program_code=program_code,
        program_name=program_name, draft_type=DRAFT_TYPE,
        draft_id=draft_id, draft_saved=draft_saved, payload=payload,
        context_pack_summary=context_pack_summary,
        source_summary=OutcomeSourceSummary(
            contexts_count=total_contexts, documents_used=all_doc_ids,
            tasks_executed=["outcome_update"], latency_ms=elapsed_ms,
        ),
        generation_status=generation_status, warnings=generation_warnings,
    )
