"""
Objective Update Service — Orchestrate R6.1B objective update draft generation.

Context pack (R6.1A) → ObjectiveUpdateSkill → optionally save draft.

Guards:
  - No writes to Program / ProgramVersion / ProgramVersionRevision.
  - Draft saved only to ctdt_analysis_drafts with draft_type="objective_update".
  - No raw file processing, no clean/chunk/embed.
  - Fail-open: LLM errors → return payload with warnings, never crash.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ctdt_objective_context_service import (
    build_objective_update_context_pack,
)

logger = logging.getLogger(__name__)

DRAFT_TYPE = "objective_update"


# ── Result DTO ───────────────────────────────────────────────────────


@dataclass
class ObjectiveSourceSummary:
    """Summary of sources used."""
    contexts_count: int
    documents_used: list[int]
    tasks_executed: list[str]
    latency_ms: int


@dataclass
class ObjectiveDraftResult:
    """Full result from generate_objective_update_draft."""
    update_cycle_id: str
    program_code: str | None
    program_name: str | None
    draft_type: str
    draft_id: int | None
    draft_saved: bool
    payload: dict[str, Any]
    context_pack_summary: dict[str, Any]
    source_summary: ObjectiveSourceSummary
    generation_status: str = "needs_generation"
    warnings: list[str] | None = None
    # R6.5: Adapted flat output for Laravel UI
    general_objective: str = ""
    specific_objectives: list | None = None
    source_summary_flat: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None
    # R6.5C: Structured objective fields
    objective_count: int = 6
    format_profile: str = ""
    general_objective_text: str = ""
    specific_objective_texts: list[str] | None = None
    quality_level: str = "good"
    quality_messages: list[str] | None = None



# ── Main orchestrator ────────────────────────────────────────────────


async def generate_objective_update_draft(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int,
    update_cycle_id: str,
    program_id: str | None = None,
    program_code: str | None = None,
    program_name: str | None = None,
    top_k_per_role: int = 5,
    user_instruction: str | None = None,
    save_draft: bool = False,
    debug_context: bool = False,
    query_svc: Any = None,
    objective_count: int = 6,
) -> ObjectiveDraftResult:
    """
    Orchestrate objective update draft generation.

    Steps:
      1. Build context pack (R6.1A).
      2. Call ObjectiveUpdateSkill with context pack.
      3. Merge missing_information from context pack.
      4. Optionally save draft.
    """
    t0 = time.monotonic()

    # ── Step 1: Build context pack ───────────────────────────────
    context_pack = await build_objective_update_context_pack(
        db,
        tenant_id=tenant_id,
        user_id=user_id,
        update_cycle_id=update_cycle_id,
        program_id=program_id,
        program_code=program_code,
        program_name=program_name,
        top_k_per_role=top_k_per_role,
        query_svc=query_svc,
    )

    # Total contexts for source summary
    total_contexts = (
        len(context_pack.current_objective_contexts)
        + len(context_pack.direction_contexts)
        + len(context_pack.legal_contexts)
        + len(context_pack.evidence_contexts)
        + len(context_pack.comparison_contexts)
    )
    all_doc_ids = context_pack.source_summary.documents_used if context_pack.source_summary else []

    # ── Step 2: Call skill ───────────────────────────────────────
    from app.services.ctdt_skills.objective_update_skill import (
        ObjectiveUpdateSkill,
        ObjectiveUpdateResult,
        ObjectiveUpdatePayload,
        ObjectiveUpdateStatus,
    )

    skill = ObjectiveUpdateSkill()

    # R6.5C: clamp objective_count
    objective_count = max(4, min(8, objective_count))

    try:
        skill_result: ObjectiveUpdateResult = await skill.run(
            update_cycle_id=update_cycle_id,
            program_id=program_id,
            program_code=program_code,
            program_name=program_name,
            context_pack=context_pack,
            user_instruction=user_instruction,
            objective_count=objective_count,
        )
    except Exception:
        logger.exception(
            "objective_update.skill_failed update_cycle=%s",
            update_cycle_id,
        )
        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.FAILED,
            payload=ObjectiveUpdatePayload(
                missing_information=list(context_pack.missing_information),
            ),
            warnings=["Skill xử lý thất bại."],
        )

    # ── Step 3: Merge context pack missing info ──────────────────
    existing_types = {
        mi.get("type") for mi in skill_result.payload.missing_information
    }
    for mi in context_pack.missing_information:
        if mi["type"] not in existing_types:
            skill_result.payload.missing_information.append(mi)

    # ── Build payload dict ───────────────────────────────────────
    payload = asdict(skill_result.payload)

    generation_status = skill_result.status
    generation_warnings = list(skill_result.warnings)

    # ── Context pack summary (role coverage + missing) ───────────
    context_pack_summary: dict[str, Any] = {
        "role_coverage": {},
        "missing_information": list(context_pack.missing_information),
    }
    for key, cov in context_pack.role_coverage.items():
        context_pack_summary["role_coverage"][key] = {
            "document_roles": cov.document_roles,
            "context_count": cov.context_count,
            "documents_used": cov.documents_used,
            "status": cov.status,
            "scoped_document_count": cov.scoped_document_count,
            "retrieval_status": cov.retrieval_status,
        }

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    # ── Step 4: Adapter + Quality Check (R6.5) ───────────────────
    # Run BEFORE save so _flat can be persisted into result_payload.
    from app.services.ctdt_objective_quality_service import (
        adapt_objective_payload,
        build_debug_context,
        check_objective_quality,
    )

    adapted = adapt_objective_payload(
        payload=payload,
        context_pack_summary=context_pack_summary,
        program_name=program_name,
        program_code=program_code,
        generation_status=generation_status,
        extra_warnings=generation_warnings,
        objective_count=objective_count,
    )

    has_evidence = bool(context_pack.evidence_contexts)
    has_current = bool(context_pack.current_objective_contexts)
    quality_warnings = check_objective_quality(
        general_objective=adapted.general_objective,
        specific_objectives=adapted.specific_objectives,
        program_name=program_name,
        has_evidence_context=has_evidence,
        has_current_curriculum_context=has_current,
        objective_count=objective_count,
    )
    all_warnings = adapted.warnings + quality_warnings

    # R6.5C: Compute quality_level
    quality_level = "good"
    quality_messages: list[str] = []
    if not has_evidence or not has_current:
        quality_level = "warning"
        quality_messages.append(
            "Nguồn minh chứng chưa đủ mạnh, nội dung cần được rà soát thủ công."
        )
    if quality_warnings:
        quality_level = "warning"
        quality_messages.extend(quality_warnings)

    # R6.5C-HARDEN-1: Check evidence_quality from skill payload
    evidence_quality = payload.get("evidence_quality", "moderate")
    if evidence_quality == "weak" and quality_level != "warning":
        quality_level = "warning"
    if evidence_quality == "weak":
        _weak_msg = (
            "Nguồn minh chứng chưa đủ mạnh, "
            "nội dung cần được rà soát thủ công."
        )
        has_similar = any(
            "minh chứng" in m and "rà soát" in m
            for m in quality_messages
        )
        if not has_similar:
            quality_messages.append(_weak_msg)

    # R6.5C: Build specific_objective_texts from adapted specific_objectives
    specific_objective_texts: list[str] = []
    for so in (adapted.specific_objectives or []):
        if isinstance(so, dict):
            specific_objective_texts.append(
                f"{so.get('code', '')}. {so.get('text', '')}"
            )
        elif isinstance(so, str):
            specific_objective_texts.append(so)

    # Build _flat block for DB persistence (no debug, no used_chunks)
    flat_block: dict[str, Any] = {
        "general_objective": adapted.general_objective,
        "general_objective_text": adapted.general_objective,
        "specific_objectives": adapted.specific_objectives,
        "specific_objective_texts": specific_objective_texts,
        "objective_count": objective_count,
        "format_profile": adapted.source_summary.get("format_profile", ""),
        "source_summary": adapted.source_summary,
        "warnings": all_warnings,
        "quality_level": quality_level,
        "quality_messages": quality_messages,
        "evidence_quality": evidence_quality,
    }

    # ── Step 5: Optional draft save ──────────────────────────────
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
                result_payload={
                    **payload,
                    "_meta": {
                        "generation_status": generation_status,
                        "warnings": generation_warnings,
                        "objective_count": objective_count,
                        "format_profile": adapted.source_summary.get("format_profile", ""),
                    },
                    "_flat": flat_block,
                },
                source_summary={
                    "contexts_count": total_contexts,
                    "documents_used": all_doc_ids,
                    "tasks_executed": ["objective_update"],
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
                "objective_update.saved draft_id=%d update_cycle=%s",
                draft_id, update_cycle_id,
            )
        except Exception:
            logger.exception(
                "objective_update.save_failed update_cycle=%s",
                update_cycle_id,
            )
            try:
                await db.rollback()
            except Exception:
                logger.exception(
                    "objective_update.rollback_failed update_cycle=%s",
                    update_cycle_id,
                )
            raise

    logger.info(
        "objective_update.done update_cycle=%s program=%s "
        "contexts=%d documents=%d elapsed_ms=%d saved=%s",
        update_cycle_id, program_code,
        total_contexts, len(all_doc_ids), elapsed_ms, draft_saved,
    )

    # ── Step 6: Debug context (never stored in DB) ───────────────
    debug_info = None
    if debug_context:
        debug_info = build_debug_context(
            context_pack=context_pack,
            queries_used=getattr(context_pack, "queries_used", []),
            fallback_used=getattr(context_pack, "fallback_used", False),
        )

    return ObjectiveDraftResult(
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        program_name=program_name,
        draft_type=DRAFT_TYPE,
        draft_id=draft_id,
        draft_saved=draft_saved,
        payload=payload,
        context_pack_summary=context_pack_summary,
        source_summary=ObjectiveSourceSummary(
            contexts_count=total_contexts,
            documents_used=all_doc_ids,
            tasks_executed=["objective_update"],
            latency_ms=elapsed_ms,
        ),
        generation_status=generation_status,
        warnings=all_warnings,
        general_objective=adapted.general_objective,
        specific_objectives=adapted.specific_objectives,
        source_summary_flat=adapted.source_summary,
        debug=debug_info,
        # R6.5C fields
        objective_count=objective_count,
        format_profile=adapted.source_summary.get("format_profile", ""),
        general_objective_text=adapted.general_objective,
        specific_objective_texts=specific_objective_texts,
        quality_level=quality_level,
        quality_messages=quality_messages,
    )
