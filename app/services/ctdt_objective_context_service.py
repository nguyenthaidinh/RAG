"""
Objective Update Context Pack Service — R6.1A.

Gom ngữ cảnh phục vụ cập nhật mục tiêu đào tạo.
Retrieve multi-role, phân nhóm context, kiểm tra role coverage.

Đây là lớp nền cho R6.1B Objective Update Skill.

Guards:
  - Không gọi LLM.
  - Không sinh mục tiêu mới.
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


# ── Role group definitions ───────────────────────────────────────────

ROLE_GROUPS: list[dict[str, Any]] = [
    {
        "key": "current_objective",
        "document_roles": ["current_curriculum"],
        "query": (
            "mục tiêu đào tạo hiện hành, mục tiêu chung, mục tiêu cụ thể, "
            "triết lý đào tạo, định hướng đào tạo trong chương trình hiện hành"
        ),
    },
    {
        "key": "direction",
        "document_roles": ["direction_decision"],
        "query": (
            "yêu cầu cập nhật chương trình đào tạo, chỉ đạo của nhà trường, "
            "định hướng cập nhật, yêu cầu đổi mới chương trình"
        ),
    },
    {
        "key": "legal",
        "document_roles": ["legal_regulation"],
        "query": (
            "quy định pháp lý, chuẩn đào tạo, yêu cầu về mục tiêu đào tạo, "
            "chuẩn chương trình đào tạo"
        ),
    },
    {
        "key": "evidence",
        "document_roles": ["survey_evidence", "meeting_report"],
        "query": (
            "khảo sát bên liên quan, phản hồi nhà tuyển dụng, sinh viên, "
            "cựu sinh viên, họp hội đồng, góp ý về mục tiêu đào tạo"
        ),
    },
    {
        "key": "comparison",
        "document_roles": ["comparison_report"],
        "query": (
            "đối sánh chương trình đào tạo, mục tiêu đào tạo của chương trình "
            "tham khảo, khoảng cách so với chương trình hiện hành"
        ),
    },
]

# Missing information rules per role group key — status="missing" (no documents)
_MISSING_RULES: dict[str, dict[str, str]] = {
    "current_objective": {
        "type": "current_objectives",
        "description": (
            "Không tìm thấy tài liệu CTĐT hiện hành chứa mục tiêu đào tạo "
            "trong phạm vi đợt cập nhật."
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
            "Không tìm thấy văn bản pháp lý/quy định về chuẩn đào tạo "
            "trong phạm vi đợt cập nhật."
        ),
    },
    "evidence": {
        "type": "survey_evidence",
        "description": (
            "Không tìm thấy kết quả khảo sát hoặc biên bản họp "
            "liên quan đến mục tiêu đào tạo."
        ),
    },
    "comparison": {
        "type": "comparison_report",
        "description": (
            "Không tìm thấy báo cáo đối sánh chương trình đào tạo "
            "trong phạm vi đợt cập nhật."
        ),
    },
}

# Missing info when status="document_available_no_context" — documents exist but
# retrieval found no relevant chunks
_CONTEXT_NOT_FOUND_RULES: dict[str, dict[str, str]] = {
    "current_objective": {
        "type": "current_objectives_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến mục tiêu đào tạo. Cần kiểm tra chunk/retrieval hoặc tài liệu gốc."
        ),
    },
    "direction": {
        "type": "direction_decision_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến chỉ đạo cập nhật CTĐT. Cần kiểm tra chunk/retrieval hoặc tài liệu gốc."
        ),
    },
    "legal": {
        "type": "legal_regulation_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến quy định pháp lý. Cần kiểm tra chunk/retrieval hoặc tài liệu gốc."
        ),
    },
    "evidence": {
        "type": "survey_evidence_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến khảo sát/biên bản họp. Cần kiểm tra chunk/retrieval hoặc tài liệu gốc."
        ),
    },
    "comparison": {
        "type": "comparison_report_context_not_found",
        "description": (
            "Có tài liệu trong phạm vi nhưng chưa tìm thấy đoạn liên quan "
            "đến đối sánh CTĐT. Cần kiểm tra chunk/retrieval hoặc tài liệu gốc."
        ),
    },
}


# ── Result DTOs ──────────────────────────────────────────────────────


@dataclass
class ContextItem:
    """A single context item from retrieval."""
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
class RoleCoverageItem:
    """Coverage status for a document role group."""
    document_roles: list[str]
    context_count: int
    documents_used: list[int]
    status: str  # "available" | "document_available_no_context" | "missing" | "failed"
    scoped_document_count: int = 0
    retrieval_status: str = "ok"  # "ok" | "failed"


@dataclass
class ContextPackSourceSummary:
    """Aggregated source summary across all role groups."""
    total_contexts: int
    documents_used: list[int]
    role_groups_retrieved: list[str]
    latency_ms: int


@dataclass
class ObjectiveUpdateContextPack:
    """Full context pack for objective update."""
    update_cycle_id: str
    program_code: str | None
    program_name: str | None
    context_pack_type: str = "objective_update"
    role_coverage: dict[str, RoleCoverageItem] = field(default_factory=dict)
    current_objective_contexts: list[ContextItem] = field(default_factory=list)
    direction_contexts: list[ContextItem] = field(default_factory=list)
    legal_contexts: list[ContextItem] = field(default_factory=list)
    evidence_contexts: list[ContextItem] = field(default_factory=list)
    comparison_contexts: list[ContextItem] = field(default_factory=list)
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
    "current_objective": "current_objective_contexts",
    "direction": "direction_contexts",
    "legal": "legal_contexts",
    "evidence": "evidence_contexts",
    "comparison": "comparison_contexts",
}


# ── Main service ─────────────────────────────────────────────────────


async def build_objective_update_context_pack(
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
) -> ObjectiveUpdateContextPack:
    """
    Build a context pack for objective update by retrieving from
    multiple document role groups.

    No LLM calls. No file processing. No DB writes.
    """
    t0 = time.monotonic()

    pack = ObjectiveUpdateContextPack(
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        program_name=program_name,
    )

    all_doc_ids: set[int] = set()
    total_contexts = 0
    groups_retrieved: list[str] = []

    for group in ROLE_GROUPS:
        key = group["key"]
        roles = group["document_roles"]
        query = group["query"]
        ctx_attr = _CONTEXT_KEY_MAP.get(key)

        logger.info(
            "objective_context.retrieve_start update_cycle=%s group=%s roles=%s",
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
                task_type=CTDTTaskType.OBJECTIVE_SUGGESTION,
                document_roles=roles,  # explicit override
                top_k=top_k_per_role,
                query_svc=query_svc,
            )
        except Exception:
            logger.exception(
                "objective_context.retrieve_failed update_cycle=%s group=%s",
                update_cycle_id, key,
            )
            retrieval_failed = True
            result = CTDTRetrievalResult(
                query=query,
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                task_type=CTDTTaskType.OBJECTIVE_SUGGESTION.value,
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
            "objective_context.retrieve_done update_cycle=%s group=%s "
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
        "objective_context.done update_cycle=%s program=%s "
        "total_contexts=%d documents=%d groups=%d missing=%d elapsed_ms=%d",
        update_cycle_id, program_code,
        total_contexts, len(all_doc_ids), len(groups_retrieved),
        len(pack.missing_information), elapsed_ms,
    )

    return pack
