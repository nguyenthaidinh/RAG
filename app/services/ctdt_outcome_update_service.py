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
    # R6.8A-PATCH-1
    objective_source: str = "none"
    quality_level: str = "good"
    quality_messages: list[str] | None = None
    outcomes_structured: list[dict] | None = None
    outcome_texts: list[str] | None = None
    outcomes_flat: list[dict] | None = None
    outcome_count: int = 10
    group_allocation: dict[str, int] | None = None


async def generate_outcome_update_draft(
    db: AsyncSession, *, tenant_id: str, user_id: int, update_cycle_id: str,
    program_id: str | None = None, program_code: str | None = None,
    program_name: str | None = None, top_k_per_role: int = 5,
    user_instruction: str | None = None, save_draft: bool = False,
    query_svc: Any = None,
    outcome_count: int = 10,
    group_allocation: dict[str, int] | None = None,
    approved_objectives: list[dict] | None = None,
    approved_objective_snapshot: dict[str, Any] | None = None,
) -> OutcomeDraftResult:
    t0 = time.monotonic()

    context_pack = await build_outcome_update_context_pack(
        db, tenant_id=tenant_id, user_id=user_id,
        update_cycle_id=update_cycle_id, program_id=program_id,
        program_code=program_code, program_name=program_name,
        top_k_per_role=top_k_per_role, query_svc=query_svc,
        approved_objectives=approved_objectives,
        approved_objective_snapshot=approved_objective_snapshot,
    )

    # R6.8A-PATCH-1 FIX 6: objective_source determined by context service
    objective_source = context_pack.objective_source

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
        _compute_default_allocation,
    )

    skill = OutcomeUpdateSkill()
    try:
        skill_result: OutcomeUpdateResult = await skill.run(
            update_cycle_id=update_cycle_id, program_id=program_id,
            program_code=program_code, program_name=program_name,
            context_pack=context_pack, user_instruction=user_instruction,
            outcome_count=outcome_count, group_allocation=group_allocation,
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
    for warning in getattr(context_pack, "objective_warnings", []):
        if warning not in generation_warnings:
            generation_warnings.append(warning)

    # R6.8A-PATCH-1 FIX 6: source-based warnings
    if objective_source == "rag_latest_objective_draft_fallback":
        generation_warnings.append(
            "Không nhận được mục tiêu đã duyệt từ Laravel; đang sử dụng bản nháp AI "
            "mục tiêu gần nhất làm căn cứ sinh chuẩn đầu ra."
        )
    elif objective_source == "legacy_laravel_approved_objectives":
        generation_warnings.append(
            "Đang sử dụng định dạng mục tiêu legacy từ Laravel; "
            "nên gửi approved_objective_snapshot đã hoàn thành "
            "để bảo đảm căn cứ sinh chuẩn đầu ra đầy đủ."
        )
    elif objective_source == "none":
        generation_warnings.append(
            "Chưa có mục tiêu đào tạo đã duyệt làm căn cứ sinh chuẩn đầu ra."
        )

    # R6.8A-PATCH-1 FIX 7: quality gate
    proposed_outcomes = payload.get("proposed_outcomes", [])
    actual_count = len(proposed_outcomes)
    effective_allocation = group_allocation or _compute_default_allocation(outcome_count)
    has_truncation = any(
        "chỉ giữ" in warning and "item đầu tiên" in warning
        for warning in generation_warnings
    )
    has_code_normalization = any(
        "chuẩn hóa mã" in warning and "C1..Cn" in warning
        for warning in generation_warnings
    )
    has_group_normalization = any(
        "chuẩn hóa nhóm" in warning
        for warning in generation_warnings
    )
    has_legacy_source = objective_source == "legacy_laravel_approved_objectives"
    has_rag_fallback_source = objective_source == "rag_latest_objective_draft_fallback"
    has_no_objective_source = objective_source == "none"
    has_flat_adapter_warning = any(
        "_flat/draft nội bộ" in warning or "M1..Mn" in warning
        for warning in generation_warnings
    )
    short_outcome_count = sum(
        1
        for po in proposed_outcomes
        if not isinstance(po.get("proposed_content"), str)
        or len(po.get("proposed_content", "").strip()) < 30
    )
    warning_flags_by_item: list[set[str]] = []
    for po in proposed_outcomes:
        raw_flags = po.get("quality_flags", [])
        if isinstance(raw_flags, list):
            warning_flags_by_item.append({str(flag).strip() for flag in raw_flags if flag})
        else:
            warning_flags_by_item.append(set())

    missing_evidence_count = sum(1 for flags in warning_flags_by_item if "missing_evidence" in flags)
    missing_mapping_count = sum(1 for flags in warning_flags_by_item if "missing_objective_mapping" in flags)
    broad_or_overlap_count = sum(
        1
        for flags in warning_flags_by_item
        if "too_broad" in flags or "overlaps_with_objective" in flags
    )
    course_specific_count = sum(1 for flags in warning_flags_by_item if "too_course_specific" in flags)
    human_review_count = sum(1 for flags in warning_flags_by_item if "needs_human_review" in flags)
    low_confidence_count = sum(
        1
        for po in proposed_outcomes
        if str(po.get("confidence") or "").strip().lower() == "low"
    )
    low_confidence_only_count = sum(
        1
        for po, flags in zip(proposed_outcomes, warning_flags_by_item)
        if str(po.get("confidence") or "").strip().lower() == "low"
        and not flags.intersection({
            "missing_evidence",
            "missing_objective_mapping",
            "too_broad",
            "overlaps_with_objective",
            "too_course_specific",
            "needs_human_review",
        })
    )
    has_item_quality_warning = any((
        missing_evidence_count,
        missing_mapping_count,
        broad_or_overlap_count,
        course_specific_count,
        human_review_count,
        low_confidence_count,
    ))

    has_generation_failure = generation_status == OutcomeUpdateStatus.FAILED or actual_count < outcome_count

    if has_generation_failure or short_outcome_count:
        quality_level = "failed"
    elif (
        has_truncation
        or has_code_normalization
        or has_group_normalization
        or has_legacy_source
        or has_rag_fallback_source
        or has_no_objective_source
        or has_flat_adapter_warning
        or has_item_quality_warning
    ):
        quality_level = "warning"
    else:
        quality_level = "good"

    quality_messages: list[str] = []
    def _add_quality_message(message: str) -> None:
        if message not in quality_messages:
            quality_messages.append(message)

    if has_generation_failure:
        _add_quality_message(
            f"AI chỉ sinh được {actual_count}/{outcome_count} chuẩn đầu ra hợp lệ."
        )
    if short_outcome_count:
        _add_quality_message(
            f"Có {short_outcome_count} chuẩn đầu ra quá ngắn (dưới 30 ký tự), chưa đủ điều kiện đưa vào biểu mẫu CTĐT."
        )
    if has_truncation:
        _add_quality_message(
            "AI sinh dư chuẩn đầu ra; hệ thống chỉ giữ đúng số lượng theo cấu trúc đã chọn."
        )
    if has_code_normalization:
        _add_quality_message("Mã CĐR đã được chuẩn hóa về C1..Cn theo cấu trúc CTĐT.")
    if has_group_normalization:
        _add_quality_message("Nhóm CĐR đã được chuẩn hóa theo phân bổ CTĐT bắt buộc.")
    if has_legacy_source:
        _add_quality_message(
            "Đang sử dụng định dạng mục tiêu legacy từ Laravel; nên gửi approved_objective_snapshot đã hoàn thành."
        )
    if has_rag_fallback_source:
        _add_quality_message(
            "Đang sử dụng bản nháp mục tiêu nội bộ thay cho snapshot mục tiêu đã duyệt từ Laravel."
        )
    if has_no_objective_source:
        _add_quality_message("Chưa có mục tiêu đào tạo đã duyệt làm căn cứ sinh chuẩn đầu ra.")
    if has_flat_adapter_warning:
        _add_quality_message(
            "Mục tiêu từ draft nội bộ đã được adapter từ định dạng _flat để sinh chuẩn đầu ra."
        )
    if missing_evidence_count:
        _add_quality_message(
            f"Có {missing_evidence_count} chuẩn đầu ra chưa có minh chứng truy xuất đủ rõ; cần rà soát nguồn trước khi sử dụng."
        )
    if missing_mapping_count:
        _add_quality_message(
            f"Có {missing_mapping_count} chuẩn đầu ra chưa liên kết rõ với mục tiêu đào tạo đã duyệt."
        )
    if broad_or_overlap_count:
        _add_quality_message(
            f"Có {broad_or_overlap_count} chuẩn đầu ra còn quá khái quát hoặc trùng vai trò với mục tiêu đào tạo."
        )
    if course_specific_count:
        _add_quality_message(
            f"Có {course_specific_count} chuẩn đầu ra đang quá chi tiết theo học phần/công cụ cụ thể."
        )
    if human_review_count:
        _add_quality_message(
            f"Có {human_review_count} chuẩn đầu ra cần cán bộ chuyên môn rà soát trước khi sử dụng."
        )
    if low_confidence_only_count:
        _add_quality_message(
            f"Có {low_confidence_only_count} chuẩn đầu ra có độ tin cậy thấp; cần rà soát trước khi sử dụng."
        )

    # R6.8A-PATCH-1 FIX 7: build structured outputs
    outcomes_structured: list[dict] = []
    outcome_texts: list[str] = []
    outcomes_flat: list[dict] = []
    for po in proposed_outcomes:
        code = po.get("code", "")
        content = po.get("proposed_content", "")
        group = po.get("outcome_type", "other")
        bloom = po.get("bloom_level", "unknown")
        mapped_codes = []
        for m in po.get("mapped_objectives", []):
            if isinstance(m, dict):
                mc = m.get("objective_code", "")
            elif isinstance(m, str):
                mc = m
            else:
                continue
            if mc:
                mapped_codes.append(mc)
        outcomes_structured.append({"code": code, "group": group, "text": content})
        outcome_texts.append(f"{code}. {content}")
        outcomes_flat.append({
            "code": code, "content": content, "group": group,
            "bloom_level": bloom, "mapped_objectives_codes": mapped_codes,
        })

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
        # R6.8A-PATCH-1 FIX 8: persist _flat metadata
        flat_block = {
            "outcome_count": outcome_count,
            "group_allocation": effective_allocation,
            "format_profile": "tay_nguyen_mau_07",
            "outcomes_structured": outcomes_structured,
            "outcome_texts": outcome_texts,
            "outcomes_flat": outcomes_flat,
            "objective_source": objective_source,
            "quality_level": quality_level,
            "quality_messages": quality_messages,
            "warnings": generation_warnings,
        }
        from app.db.models.ctdt_analysis_draft import CTDTAnalysisDraft
        try:
            draft = CTDTAnalysisDraft(
                tenant_id=tenant_id, update_cycle_id=update_cycle_id,
                program_id=program_id, program_code=program_code,
                program_name=program_name, analysis_mode="design",
                draft_type=DRAFT_TYPE,
                result_payload={
                    **payload,
                    "_flat": flat_block,
                    "_meta": {"generation_status": generation_status, "warnings": generation_warnings},
                },
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
        # R6.8A-PATCH-1 additions
        objective_source=objective_source,
        quality_level=quality_level,
        quality_messages=quality_messages,
        outcomes_structured=outcomes_structured,
        outcome_texts=outcome_texts,
        outcomes_flat=outcomes_flat,
        outcome_count=outcome_count,
        group_allocation=effective_allocation,
    )
