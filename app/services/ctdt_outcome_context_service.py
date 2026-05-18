"""
Outcome Update Context Pack Service — R6.2A.

Gom ngữ cảnh phục vụ cập nhật chuẩn đầu ra (CĐR/PLO).
Retrieve multi-role, phân nhóm context, đọc latest objective_update draft,
kiểm tra role coverage.

Đây là lớp nền cho R6.2B Outcome Update Skill.

Guards:
  - Không gọi LLM.
  - Không sinh chuẩn đầu ra mới.
  - Không xử lý file thô / clean / chunk / embed lại.
  - Không ghi Program / ProgramVersion / ProgramVersionRevision.
  - Chỉ dùng contexts đã qua pipeline RAG.
  - Không query global ngoài scope update_cycle/program.
  - Thiếu role → missing_information, không bịa.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ctdt_retrieval_service import (
    CTDTRetrievalContext,
    CTDTRetrievalResult,
    CTDTTaskType,
    ctdt_retrieve,
)

logger = logging.getLogger(__name__)


# ── Re-use DTOs from R6.1A ──────────────────────────────────────────

from app.services.ctdt_objective_context_service import (  # noqa: E402
    ContextItem,
    ContextPackSourceSummary,
    RoleCoverageItem,
)

# Re-export for tests / downstream
__all__ = [
    "ContextItem",
    "ContextPackSourceSummary",
    "RoleCoverageItem",
    "OutcomeUpdateContextPack",
    "build_outcome_update_context_pack",
]


# ── Role group definitions (outcome-specific) ───────────────────────

ROLE_GROUPS: list[dict[str, Any]] = [
    {
        "key": "current_outcome",
        "document_roles": ["current_curriculum"],
        "query": (
            "chuẩn đầu ra hiện hành, PLO, CĐR, kết quả học tập mong đợi, "
            "năng lực người học sau tốt nghiệp"
        ),
    },
    {
        "key": "current_curriculum",
        "document_roles": ["current_curriculum"],
        "query": (
            "cấu trúc chương trình hiện hành, mục tiêu đào tạo, chuẩn đầu ra, "
            "học phần liên quan đến chuẩn đầu ra"
        ),
    },
    {
        "key": "direction",
        "document_roles": ["direction_decision"],
        "query": (
            "yêu cầu cập nhật chuẩn đầu ra, định hướng cập nhật năng lực "
            "người học, yêu cầu đổi mới chương trình"
        ),
    },
    {
        "key": "legal",
        "document_roles": ["legal_regulation"],
        "query": (
            "quy định pháp lý về chuẩn đầu ra, chuẩn chương trình đào tạo, "
            "yêu cầu năng lực người học"
        ),
    },
    {
        "key": "evidence",
        "document_roles": ["survey_evidence", "meeting_report"],
        "query": (
            "khảo sát bên liên quan, nhà tuyển dụng, sinh viên, cựu sinh viên, "
            "hội đồng góp ý về chuẩn đầu ra, năng lực cần bổ sung"
        ),
    },
    {
        "key": "comparison",
        "document_roles": ["comparison_report"],
        "query": (
            "đối sánh chuẩn đầu ra, PLO của chương trình tham khảo, "
            "khoảng cách so với chuẩn đầu ra hiện hành"
        ),
    },
    {
        "key": "course_syllabus",
        "document_roles": ["course_syllabus"],
        "query": (
            "đề cương học phần, học phần hỗ trợ chuẩn đầu ra, "
            "năng lực/học phần liên quan đến CĐR"
        ),
    },
]


# ── Missing information rules ────────────────────────────────────────

_MISSING_RULES: dict[str, dict[str, str]] = {
    "current_outcome": {
        "type": "current_outcomes",
        "description": (
            "Không tìm thấy tài liệu CTĐT hiện hành chứa chuẩn đầu ra "
            "trong phạm vi đợt cập nhật."
        ),
    },
    "current_curriculum": {
        "type": "current_curriculum",
        "description": (
            "Không tìm thấy tài liệu CTĐT hiện hành chứa cấu trúc "
            "chương trình trong phạm vi đợt cập nhật."
        ),
    },
    "direction": {
        "type": "direction_decision",
        "description": (
            "Không tìm thấy quyết định/chỉ đạo cập nhật CTĐT "
            "trong phạm vi đợt cập nhật."
        ),
    },
    "legal": {
        "type": "legal_regulation",
        "description": (
            "Không tìm thấy văn bản pháp lý/quy định về chuẩn đầu ra "
            "trong phạm vi đợt cập nhật."
        ),
    },
    "evidence": {
        "type": "survey_evidence",
        "description": (
            "Không tìm thấy kết quả khảo sát hoặc biên bản họp "
            "liên quan đến chuẩn đầu ra."
        ),
    },
    "comparison": {
        "type": "comparison_report",
        "description": (
            "Không tìm thấy báo cáo đối sánh chuẩn đầu ra "
            "trong phạm vi đợt cập nhật."
        ),
    },
    "course_syllabus": {
        "type": "course_syllabus",
        "description": (
            "Không tìm thấy đề cương học phần "
            "trong phạm vi đợt cập nhật."
        ),
    },
}

_CONTEXT_NOT_FOUND_RULES: dict[str, dict[str, str]] = {
    "current_outcome": {
        "type": "current_outcomes_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến chuẩn đầu ra. Cần kiểm tra chunk/retrieval hoặc tài liệu gốc."
        ),
    },
    "current_curriculum": {
        "type": "current_curriculum_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến cấu trúc chương trình. Cần kiểm tra chunk/retrieval."
        ),
    },
    "direction": {
        "type": "direction_decision_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến chỉ đạo cập nhật. Cần kiểm tra chunk/retrieval."
        ),
    },
    "legal": {
        "type": "legal_regulation_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến quy định pháp lý. Cần kiểm tra chunk/retrieval."
        ),
    },
    "evidence": {
        "type": "survey_evidence_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến khảo sát/biên bản họp. Cần kiểm tra chunk/retrieval."
        ),
    },
    "comparison": {
        "type": "comparison_report_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến đối sánh CĐR. Cần kiểm tra chunk/retrieval."
        ),
    },
    "course_syllabus": {
        "type": "course_syllabus_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến đề cương học phần. Cần kiểm tra chunk/retrieval."
        ),
    },
}


# ── Result DTO ───────────────────────────────────────────────────────


@dataclass
class OutcomeUpdateContextPack:
    """Full context pack for outcome (CĐR) update."""
    update_cycle_id: str
    program_code: str | None
    program_name: str | None
    context_pack_type: str = "outcome_update"
    role_coverage: dict[str, RoleCoverageItem] = field(default_factory=dict)
    # Objective update draft (from R6.1B)
    objective_update_contexts: list[ContextItem] = field(default_factory=list)
    objective_update_payload: dict[str, Any] | None = None
    # Retrieval-based groups
    current_outcome_contexts: list[ContextItem] = field(default_factory=list)
    current_curriculum_contexts: list[ContextItem] = field(default_factory=list)
    direction_contexts: list[ContextItem] = field(default_factory=list)
    legal_contexts: list[ContextItem] = field(default_factory=list)
    evidence_contexts: list[ContextItem] = field(default_factory=list)
    comparison_contexts: list[ContextItem] = field(default_factory=list)
    course_syllabus_contexts: list[ContextItem] = field(default_factory=list)
    other_contexts: list[ContextItem] = field(default_factory=list)
    missing_information: list[dict[str, str]] = field(default_factory=list)
    source_summary: ContextPackSourceSummary | None = None


# ── Helpers ──────────────────────────────────────────────────────────


def _ctx_to_item(ctx: CTDTRetrievalContext) -> ContextItem:
    """Convert a retrieval context to a ContextItem."""
    return ContextItem(
        ai_document_id=ctx.ai_document_id,
        external_file_id=ctx.external_file_id,
        filename=ctx.filename,
        document_role=ctx.document_role,
        chunk_id=ctx.chunk_id,
        chunk_index=ctx.chunk_index,
        score=round(ctx.score, 4),
        text=ctx.text,
        source=ctx.source if isinstance(ctx.source, dict) else {},
    )


_CONTEXT_KEY_MAP = {
    "current_outcome": "current_outcome_contexts",
    "current_curriculum": "current_curriculum_contexts",
    "direction": "direction_contexts",
    "legal": "legal_contexts",
    "evidence": "evidence_contexts",
    "comparison": "comparison_contexts",
    "course_syllabus": "course_syllabus_contexts",
}


# ── Latest objective_update draft reader ─────────────────────────────


async def _load_latest_objective_draft(
    db: AsyncSession,
    *,
    tenant_id: str,
    update_cycle_id: str,
    program_code: str | None,
) -> dict[str, Any] | None:
    """
    Try to load the latest objective_update draft payload.
    Returns the result_payload dict or None.
    No writes, no LLM.
    """
    try:
        from app.services.ctdt_analysis_draft_service import (
            get_latest_analysis_draft,
        )

        draft = await get_latest_analysis_draft(
            db,
            tenant_id=tenant_id,
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            analysis_mode="design",
            draft_type="objective_update",
            status="draft",
        )
        if draft is not None and draft.result_payload:
            return draft.result_payload
    except Exception:
        logger.warning(
            "outcome_context.load_objective_draft_failed "
            "update_cycle=%s program=%s",
            update_cycle_id, program_code,
            exc_info=True,
        )
    return None


# ── Main service ─────────────────────────────────────────────────────


async def build_outcome_update_context_pack(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int,
    update_cycle_id: str,
    program_id: str | None = None,
    program_code: str | None = None,
    program_name: str | None = None,
    top_k_per_role: int = 5,
    query_svc: Any = None,
) -> OutcomeUpdateContextPack:
    """
    Build a context pack for outcome (CĐR) update by:
      1. Loading latest objective_update draft.
      2. Retrieving from multiple document role groups.

    No LLM calls. No file processing. No DB writes.
    """
    t0 = time.monotonic()

    pack = OutcomeUpdateContextPack(
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        program_name=program_name,
    )

    all_doc_ids: set[int] = set()
    total_contexts = 0
    groups_retrieved: list[str] = []

    # ── Step 1: Load latest objective_update draft ────────────────
    obj_payload = await _load_latest_objective_draft(
        db,
        tenant_id=tenant_id,
        update_cycle_id=update_cycle_id,
        program_code=program_code,
    )

    if obj_payload is not None:
        pack.objective_update_payload = obj_payload
        pack.role_coverage["objective_update"] = RoleCoverageItem(
            document_roles=["objective_update_draft"],
            context_count=1,
            documents_used=[],
            status="available",
            scoped_document_count=0,
            retrieval_status="ok",
        )
        logger.info(
            "outcome_context.objective_draft_loaded update_cycle=%s",
            update_cycle_id,
        )
    else:
        pack.role_coverage["objective_update"] = RoleCoverageItem(
            document_roles=["objective_update_draft"],
            context_count=0,
            documents_used=[],
            status="missing",
            scoped_document_count=0,
            retrieval_status="ok",
        )
        pack.missing_information.append({
            "type": "objective_update",
            "description": (
                "Chưa có bản nháp mục tiêu đào tạo để làm căn cứ "
                "sinh chuẩn đầu ra."
            ),
        })

    groups_retrieved.append("objective_update")

    # ── Step 2: Role-aware retrieval ─────────────────────────────
    for group in ROLE_GROUPS:
        key = group["key"]
        roles = group["document_roles"]
        query = group["query"]
        ctx_attr = _CONTEXT_KEY_MAP.get(key)

        logger.info(
            "outcome_context.retrieve_start update_cycle=%s group=%s roles=%s",
            update_cycle_id, key, roles,
        )

        retrieval_failed = False
        try:
            result: CTDTRetrievalResult = await ctdt_retrieve(
                db,
                tenant_id=tenant_id,
                user_id=user_id,
                query=query,
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_id=program_id,
                task_type=CTDTTaskType.OUTCOME_SUGGESTION,
                document_roles=roles,  # explicit override
                top_k=top_k_per_role,
                query_svc=query_svc,
            )
        except Exception:
            logger.exception(
                "outcome_context.retrieve_failed update_cycle=%s group=%s",
                update_cycle_id, key,
            )
            retrieval_failed = True
            result = CTDTRetrievalResult(
                query=query,
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                task_type=CTDTTaskType.OUTCOME_SUGGESTION.value,
                document_roles_used=roles,
                contexts=[],
                scoped_document_count=0,
                latency_ms=0,
            )

        # Convert to context items
        items = [_ctx_to_item(ctx) for ctx in result.contexts]
        group_doc_ids = sorted({it.ai_document_id for it in items})
        scoped_doc_count = getattr(result, "scoped_document_count", 0) or 0

        # Assign to the correct attribute
        if ctx_attr:
            setattr(pack, ctx_attr, items)

        # ── Determine status ─────────────────────────────────────
        if retrieval_failed:
            group_status = "failed"
            retrieval_status = "failed"
        elif len(items) > 0:
            group_status = "available"
            retrieval_status = "ok"
        elif scoped_doc_count > 0:
            group_status = "document_available_no_context"
            retrieval_status = "ok"
        else:
            group_status = "missing"
            retrieval_status = "ok"

        pack.role_coverage[key] = RoleCoverageItem(
            document_roles=roles,
            context_count=len(items),
            documents_used=group_doc_ids,
            status=group_status,
            scoped_document_count=scoped_doc_count,
            retrieval_status=retrieval_status,
        )

        # ── Missing information rules ────────────────────────────
        if group_status == "missing" and key in _MISSING_RULES:
            pack.missing_information.append(_MISSING_RULES[key])
        elif group_status == "document_available_no_context" and key in _CONTEXT_NOT_FOUND_RULES:
            pack.missing_information.append(_CONTEXT_NOT_FOUND_RULES[key])

        # Track global stats
        all_doc_ids.update(group_doc_ids)
        total_contexts += len(items)
        groups_retrieved.append(key)

        logger.info(
            "outcome_context.retrieve_done update_cycle=%s group=%s "
            "contexts=%d docs=%d",
            update_cycle_id, key, len(items), len(group_doc_ids),
        )

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    pack.source_summary = ContextPackSourceSummary(
        total_contexts=total_contexts,
        documents_used=sorted(all_doc_ids),
        role_groups_retrieved=groups_retrieved,
        latency_ms=elapsed_ms,
    )

    logger.info(
        "outcome_context.done update_cycle=%s program=%s "
        "total_contexts=%d documents=%d groups=%d missing=%d elapsed_ms=%d",
        update_cycle_id, program_code,
        total_contexts, len(all_doc_ids), len(groups_retrieved),
        len(pack.missing_information), elapsed_ms,
    )

    return pack
