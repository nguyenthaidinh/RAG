"""
Objective Update Context Pack Service — R6.1A + R6.5 enhancements.

Gom ngữ cảnh phục vụ cập nhật mục tiêu đào tạo.
Retrieve multi-role, phân nhóm context, kiểm tra role coverage.

R6.5 enhancements:
  - Multi-query retrieval per role group (YC3).
  - Fallback khi document_role sai hoặc thiếu (YC4).
  - Deduplication theo (document_id, chunk_index).
  - Ưu tiên context có heading liên quan "mục tiêu".
  - Tracking queries_used + fallback_used.

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
# R6.5: Each group now has `queries` (list) instead of single `query`.

ROLE_GROUPS: list[dict[str, Any]] = [
    {
        "key": "current_objective",
        "document_roles": ["current_curriculum"],
        "queries": [
            (
                "mục tiêu đào tạo hiện hành, mục tiêu chung, mục tiêu cụ thể, "
                "triết lý đào tạo, định hướng đào tạo trong chương trình hiện hành"
            ),
            "mục tiêu đào tạo",
            "mục tiêu chung của chương trình đào tạo",
            "mục tiêu cụ thể chương trình đào tạo",
            "program educational objectives PEO",
        ],
        # Backward compat: keep single `query` for external callers
        "query": (
            "mục tiêu đào tạo hiện hành, mục tiêu chung, mục tiêu cụ thể, "
            "triết lý đào tạo, định hướng đào tạo trong chương trình hiện hành"
        ),
        "fallback_enabled": True,
    },
    {
        "key": "direction",
        "document_roles": ["direction_decision"],
        "queries": [
            (
                "yêu cầu cập nhật chương trình đào tạo, chỉ đạo của nhà trường, "
                "định hướng cập nhật, yêu cầu đổi mới chương trình"
            ),
        ],
        "query": (
            "yêu cầu cập nhật chương trình đào tạo, chỉ đạo của nhà trường, "
            "định hướng cập nhật, yêu cầu đổi mới chương trình"
        ),
        "fallback_enabled": False,
    },
    {
        "key": "legal",
        "document_roles": ["legal_regulation"],
        "queries": [
            (
                "quy định pháp lý, chuẩn đào tạo, yêu cầu về mục tiêu đào tạo, "
                "chuẩn chương trình đào tạo"
            ),
        ],
        "query": (
            "quy định pháp lý, chuẩn đào tạo, yêu cầu về mục tiêu đào tạo, "
            "chuẩn chương trình đào tạo"
        ),
        "fallback_enabled": False,
    },
    {
        "key": "evidence",
        "document_roles": ["survey_evidence", "meeting_report"],
        "queries": [
            (
                "khảo sát bên liên quan, phản hồi nhà tuyển dụng, sinh viên, "
                "cựu sinh viên, họp hội đồng, góp ý về mục tiêu đào tạo"
            ),
            "nhu cầu xã hội, năng lực nghề nghiệp, khảo sát bên liên quan",
        ],
        "query": (
            "khảo sát bên liên quan, phản hồi nhà tuyển dụng, sinh viên, "
            "cựu sinh viên, họp hội đồng, góp ý về mục tiêu đào tạo"
        ),
        "fallback_enabled": False,
    },
    {
        "key": "comparison",
        "document_roles": ["comparison_report"],
        "queries": [
            (
                "đối sánh chương trình đào tạo, mục tiêu đào tạo của chương trình "
                "tham khảo, khoảng cách so với chương trình hiện hành"
            ),
        ],
        "query": (
            "đối sánh chương trình đào tạo, mục tiêu đào tạo của chương trình "
            "tham khảo, khoảng cách so với chương trình hiện hành"
        ),
        "fallback_enabled": False,
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
    # R6.5 additions
    queries_used: list[str] = field(default_factory=list)
    fallback_used: bool = False


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


# ── R6.5: Heading relevance boost ───────────────────────────────────

_OBJECTIVE_HEADINGS = [
    "mục tiêu đào tạo", "mục tiêu chung", "mục tiêu cụ thể",
    "mục tiêu của chương trình", "program educational objectives",
    "training objectives", "peo",
]

_HEADING_SCORE_BOOST = 0.05  # small boost to prioritize heading-matched chunks


def _boost_heading_relevance(items: list[ContextItem]) -> list[ContextItem]:
    """Boost score for chunks containing objective-related headings."""
    for item in items:
        text_lower = (item.text or "").lower()
        if any(h in text_lower for h in _OBJECTIVE_HEADINGS):
            item.score = round(item.score + _HEADING_SCORE_BOOST, 4)
    return items


# ── R6.5: Multi-query dedup ─────────────────────────────────────────


def _deduplicate_items(items: list[ContextItem]) -> list[ContextItem]:
    """Deduplicate by (ai_document_id, chunk_index), keep highest score."""
    seen: dict[str, ContextItem] = {}
    for item in items:
        key = f"{item.ai_document_id}:{item.chunk_index}"
        existing = seen.get(key)
        if existing is None or item.score > existing.score:
            seen[key] = item
    # Sort by score desc
    return sorted(seen.values(), key=lambda x: x.score, reverse=True)


# ── R6.5: Build program-specific queries (Group C) ──────────────────


def _build_program_queries(
    program_name: str | None,
    program_code: str | None,
) -> list[str]:
    """Build program-specific queries (Nhóm C) if program info available."""
    queries: list[str] = []
    if program_name:
        queries.extend([
            f"{program_name} mục tiêu đào tạo",
            f"{program_name} chuẩn đầu ra",
            f"{program_name} vị trí việc làm",
        ])
    if program_code:
        queries.append(f"{program_code} mục tiêu đào tạo")
    return queries


# ── R6.5: Multi-query retrieve for one group ────────────────────────


async def _multi_query_retrieve(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int,
    queries: list[str],
    update_cycle_id: str,
    program_code: str | None,
    program_id: str | None,
    document_roles: list[str],
    top_k_per_query: int,
    query_svc: Any = None,
) -> tuple[list[ContextItem], int, bool]:
    """
    Run multiple queries for a single role group, deduplicate results.

    Returns: (items, scoped_document_count, retrieval_failed).
    """
    all_items: list[ContextItem] = []
    scoped_doc_count = 0
    retrieval_failed = False

    for query in queries:
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
                document_roles=document_roles,
                top_k=top_k_per_query,
                query_svc=query_svc,
            )
            items = [_ctx_to_item(ctx) for ctx in result.contexts]
            all_items.extend(items)
            scoped_doc_count = max(
                scoped_doc_count,
                getattr(result, "scoped_document_count", 0) or 0,
            )
        except Exception:
            logger.exception(
                "objective_context.multi_query_failed query=%s",
                query[:100],
            )
            retrieval_failed = True

    # Deduplicate and boost heading-relevant chunks
    deduped = _deduplicate_items(all_items)
    deduped = _boost_heading_relevance(deduped)
    # Re-sort after boost
    deduped.sort(key=lambda x: x.score, reverse=True)

    return deduped, scoped_doc_count, retrieval_failed


# ── R6.5: Fallback retrieve (no role filter) ────────────────────────


async def _fallback_retrieve(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int,
    queries: list[str],
    update_cycle_id: str,
    program_code: str | None,
    program_id: str | None,
    top_k: int,
    query_svc: Any = None,
) -> tuple[list[ContextItem], int]:
    """
    Fallback: retrieve without document_role filter, same update_cycle.

    Guard: NEVER crosses update_cycle boundary.
    Guard: NEVER crosses program boundary if program_id/program_code given.
    """
    all_items: list[ContextItem] = []
    scoped_doc_count = 0

    # Use first 2 queries max to limit cost
    for query in queries[:2]:
        try:
            result = await ctdt_retrieve(
                db,
                tenant_id=tenant_id,
                user_id=user_id,
                query=query,
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_id=program_id,
                task_type=CTDTTaskType.OBJECTIVE_SUGGESTION,
                document_roles=None,  # No role filter = fallback
                top_k=top_k,
                query_svc=query_svc,
            )
            items = [_ctx_to_item(ctx) for ctx in result.contexts]
            all_items.extend(items)
            scoped_doc_count = max(
                scoped_doc_count,
                getattr(result, "scoped_document_count", 0) or 0,
            )
        except Exception:
            logger.exception(
                "objective_context.fallback_failed query=%s",
                query[:100],
            )

    deduped = _deduplicate_items(all_items)
    deduped = _boost_heading_relevance(deduped)
    deduped.sort(key=lambda x: x.score, reverse=True)

    return deduped, scoped_doc_count


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

    R6.5: multi-query per group, dedup, heading boost, fallback.

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
    all_queries_used: list[str] = []

    # R6.5: Build program-specific queries (Nhóm C)
    program_queries = _build_program_queries(program_name, program_code)

    for group in ROLE_GROUPS:
        key = group["key"]
        roles = group["document_roles"]
        ctx_attr = _CONTEXT_KEY_MAP.get(key)

        # R6.5: Use multi-query list
        queries = list(group.get("queries", [group.get("query", "")]))

        # R6.5: Append program-specific queries for current_objective group
        if key == "current_objective" and program_queries:
            queries.extend(program_queries)

        all_queries_used.extend(queries)

        logger.info(
            "objective_context.retrieve_start update_cycle=%s group=%s "
            "roles=%s queries=%d",
            update_cycle_id, key, roles, len(queries),
        )

        # R6.5: Multi-query retrieval
        items, scoped_doc_count, retrieval_failed = await _multi_query_retrieve(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            queries=queries,
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            program_id=program_id,
            document_roles=roles,
            top_k_per_query=top_k_per_role,
            query_svc=query_svc,
        )

        # R6.5: Limit items per group to avoid context explosion
        items = items[:top_k_per_role * 2]  # Allow up to 2x since multi-query

        # R6.5: Fallback for current_objective if no results
        fallback_triggered = False
        if (
            not items
            and not retrieval_failed
            and group.get("fallback_enabled", False)
        ):
            logger.info(
                "objective_context.fallback_start update_cycle=%s group=%s",
                update_cycle_id, key,
            )
            fallback_items, fallback_doc_count = await _fallback_retrieve(
                db,
                tenant_id=tenant_id,
                user_id=user_id,
                queries=queries[:2],
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_id=program_id,
                top_k=top_k_per_role,
                query_svc=query_svc,
            )
            if fallback_items:
                items = fallback_items[:top_k_per_role]
                scoped_doc_count = max(scoped_doc_count, fallback_doc_count)
                fallback_triggered = True
                pack.fallback_used = True
                logger.info(
                    "objective_context.fallback_found update_cycle=%s group=%s "
                    "contexts=%d",
                    update_cycle_id, key, len(items),
                )

        group_doc_ids = sorted({it.ai_document_id for it in items})

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

        # R6.5: Fallback warning
        if fallback_triggered:
            pack.missing_information.append({
                "type": f"{key}_fallback_used",
                "description": (
                    "Không tìm thấy đủ ngữ cảnh trong tài liệu đúng vai trò. "
                    "Hệ thống đã mở rộng tìm kiếm trong toàn bộ tài liệu "
                    "của đợt cập nhật."
                ),
            })

        # Track global stats
        all_doc_ids.update(group_doc_ids)
        total_contexts += len(items)
        groups_retrieved.append(key)

        logger.info(
            "objective_context.retrieve_done update_cycle=%s group=%s "
            "contexts=%d docs=%d fallback=%s",
            update_cycle_id, key, len(items), len(group_doc_ids),
            fallback_triggered,
        )

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    pack.source_summary = ContextPackSourceSummary(
        total_contexts=total_contexts,
        documents_used=sorted(all_doc_ids),
        role_groups_retrieved=groups_retrieved,
        latency_ms=elapsed_ms,
    )
    pack.queries_used = all_queries_used

    logger.info(
        "objective_context.done update_cycle=%s program=%s "
        "total_contexts=%d documents=%d groups=%d missing=%d "
        "fallback=%s elapsed_ms=%d",
        update_cycle_id, program_code,
        total_contexts, len(all_doc_ids), len(groups_retrieved),
        len(pack.missing_information), pack.fallback_used, elapsed_ms,
    )

    return pack
