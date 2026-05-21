"""
Objective Quality Service — R6.5 Objective Quality Pack.

Provides:
  - adapt_objective_payload(): chuyển raw skill payload sang format phẳng cho Laravel.
  - check_objective_quality(): kiểm tra chất lượng nhẹ bằng code, trả warnings.
  - build_debug_context(): tạo debug info cho Step 2.1.

Guards:
  - Không gọi LLM.
  - Không ghi DB.
  - Không phá backward compatibility — raw_payload luôn được giữ lại.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Adapted Output DTOs ─────────────────────────────────────────────


@dataclass
class ObjectiveAdaptedResult:
    """Flat result ready for Laravel UI binding."""
    general_objective: str
    specific_objectives: list[str]
    warnings: list[str]
    source_summary: dict[str, Any]
    raw_payload: dict[str, Any]


@dataclass
class DebugChunkInfo:
    """Rút gọn info về 1 chunk dùng trong context."""
    document_id: int
    document_name: str | None
    document_role: str | None
    chunk_index: int
    score: float
    text_preview: str  # 300-500 chars


@dataclass
class DebugContext:
    """Debug info cho Step 2.1."""
    queries: list[str]
    used_chunks: list[dict[str, Any]]
    missing_roles: list[str]
    fallback_used: bool
    context_char_count: int


# ── Outcome-overlap detector (reused from skill) ────────────────────

_OUTCOME_VERBS = frozenset([
    "phân tích", "thiết kế", "triển khai", "đánh giá",
    "vận dụng", "xây dựng", "kiểm thử", "lập trình",
    "cài đặt", "mô hình hóa",
])

_GENERIC_PHRASES = [
    "có kiến thức và kỹ năng cần thiết",
    "đáp ứng nhu cầu xã hội",
    "có phẩm chất đạo đức tốt",
    "có năng lực chuyên môn",
    "đáp ứng yêu cầu",
    "phục vụ sự nghiệp",
    "góp phần phát triển",
]


# ── Response Adapter ─────────────────────────────────────────────────


def adapt_objective_payload(
    *,
    payload: dict[str, Any],
    context_pack_summary: dict[str, Any] | None = None,
    program_name: str | None = None,
    program_code: str | None = None,
    generation_status: str = "needs_generation",
    extra_warnings: list[str] | None = None,
) -> ObjectiveAdaptedResult:
    """
    Chuyển raw objective payload (từ ObjectiveUpdateSkill) sang format phẳng.

    Quy tắc:
      - Nếu payload có general_objective_text → lấy trực tiếp.
      - Nếu không, duyệt proposed_objectives → phân loại general vs specific.
      - Nếu không phân loại được → heuristic + warning.
      - raw_payload luôn giữ lại toàn bộ.
    """
    warnings: list[str] = list(extra_warnings or [])
    general_objective = ""
    specific_objectives: list[str] = []

    # ── Priority 1: Trực tiếp từ field mới (nếu prompt mới sinh ra) ──
    direct_general = (payload.get("general_objective_text") or "").strip()
    direct_specifics = payload.get("specific_objective_texts")

    if direct_general:
        general_objective = direct_general
        if isinstance(direct_specifics, list):
            specific_objectives = [
                s.strip() for s in direct_specifics
                if isinstance(s, str) and s.strip()
            ]
    else:
        # ── Priority 2: Phân loại từ proposed_objectives ─────────────
        proposed = payload.get("proposed_objectives", [])
        if isinstance(proposed, list) and proposed:
            generals: list[str] = []
            specifics: list[str] = []

            for obj in proposed:
                if not isinstance(obj, dict):
                    continue
                content = (obj.get("proposed_content") or "").strip()
                if not content:
                    continue
                obj_type = (obj.get("objective_type") or "").strip()

                if obj_type == "general_objective":
                    generals.append(content)
                elif obj_type == "specific_objective":
                    specifics.append(content)
                else:
                    # Không rõ type → thử phân loại theo heuristic
                    specifics.append(content)

            if generals:
                general_objective = " ".join(generals)
                specific_objectives = specifics
            elif specifics:
                # Không có general rõ ràng → heuristic: mục dài nhất → general
                sorted_by_len = sorted(
                    specifics, key=len, reverse=True,
                )
                general_objective = sorted_by_len[0]
                specific_objectives = sorted_by_len[1:]
                warnings.append(
                    "Không xác định rõ mục tiêu chung từ AI. "
                    "Hệ thống đã tự phân loại, cần rà soát lại."
                )
        elif generation_status == "generated":
            warnings.append(
                "AI đã xử lý nhưng không sinh được mục tiêu đào tạo cụ thể. "
                "Cần kiểm tra lại tài liệu đầu vào."
            )

    # ── Build source_summary ────────────────────────────────────────
    source_summary = _build_source_summary(
        context_pack_summary=context_pack_summary,
        program_name=program_name,
        program_code=program_code,
    )

    return ObjectiveAdaptedResult(
        general_objective=general_objective,
        specific_objectives=specific_objectives,
        warnings=warnings,
        source_summary=source_summary,
        raw_payload=payload,
    )


def _build_source_summary(
    *,
    context_pack_summary: dict[str, Any] | None,
    program_name: str | None,
    program_code: str | None,
) -> dict[str, Any]:
    """Build flat source_summary for Laravel."""
    summary: dict[str, Any] = {
        "detected_program_name": program_name or "",
        "detected_program_code": program_code or "",
        "detected_degree_level": "",
        "used_documents": [],
    }

    if not context_pack_summary:
        return summary

    # Extract document IDs from role_coverage
    role_coverage = context_pack_summary.get("role_coverage", {})
    doc_ids: set[int] = set()
    for _key, cov in role_coverage.items():
        if isinstance(cov, dict):
            for did in cov.get("documents_used", []):
                doc_ids.add(did)

    summary["used_documents"] = sorted(doc_ids)
    return summary


# ── Quality Check ────────────────────────────────────────────────────


def check_objective_quality(
    *,
    general_objective: str,
    specific_objectives: list[str],
    program_name: str | None = None,
    has_evidence_context: bool = True,
    has_current_curriculum_context: bool = True,
) -> list[str]:
    """
    Kiểm tra chất lượng nhẹ bằng code. Trả list warnings tiếng Việt.

    Không chặn response. Chỉ trả warning để UI hiển thị.
    """
    warnings: list[str] = []

    # 1. general_objective không rỗng
    if not general_objective.strip():
        warnings.append(
            "Mục tiêu chung (general objective) chưa được sinh. "
            "Cần kiểm tra tài liệu đầu vào hoặc thử lại."
        )

    # 2. specific_objectives ít nhất 4 mục
    num_specific = len(specific_objectives)
    if num_specific < 4:
        warnings.append(
            f"Chỉ có {num_specific} mục tiêu cụ thể (khuyến nghị 4-6 mục). "
            "Cần bổ sung thêm mục tiêu cụ thể."
        )

    # 3. Quá nhiều mục
    if num_specific > 6:
        warnings.append(
            f"Có {num_specific} mục tiêu cụ thể (khuyến nghị tối đa 6 mục). "
            "Cần xem xét gộp hoặc lược bớt."
        )

    # 4. Nội dung quá chung chung
    if general_objective.strip():
        generic_count = sum(
            1 for phrase in _GENERIC_PHRASES
            if phrase in general_objective.lower()
        )
        if generic_count >= 2:
            warnings.append(
                "Nội dung sinh ra còn chung chung, cần rà soát "
                "và bổ sung thông tin đặc thù của ngành."
            )

    # 5. Bám ngành
    if program_name and general_objective.strip():
        # Kiểm tra xem có dấu hiệu ngành/lĩnh vực không
        name_lower = program_name.lower()
        all_text = (
            general_objective + " " + " ".join(specific_objectives)
        ).lower()

        # Trích từ khóa từ program_name (tách bỏ stop words)
        _stop_words = {"đào", "tạo", "ngành", "chương", "trình", "cử", "nhân",
                       "kỹ", "sư", "thạc", "sĩ", "tiến", "và", "của", "trong"}
        keywords = [
            w for w in name_lower.split()
            if len(w) > 2 and w not in _stop_words
        ]

        has_match = any(kw in all_text for kw in keywords) if keywords else True
        if not has_match:
            warnings.append(
                "Không xác định được tên ngành từ tài liệu, "
                "hệ thống đã dùng thông tin truyền từ Laravel nếu có."
            )

    # 6. Thiếu evidence context
    if not has_evidence_context:
        warnings.append(
            "Không tìm thấy kết quả khảo sát hoặc góp ý bên liên quan "
            "trong tài liệu đã cung cấp."
        )

    # 7. Thiếu CTĐT hiện hành
    if not has_current_curriculum_context:
        warnings.append(
            "Không tìm thấy rõ đoạn mục tiêu đào tạo hiện hành "
            "trong tài liệu đã cung cấp."
        )

    # 8. Specific objectives quá giống CĐR
    outcome_like_count = 0
    for spec in specific_objectives:
        spec_lower = spec.lower()
        verb_hits = sum(1 for v in _OUTCOME_VERBS if v in spec_lower)
        if verb_hits >= 3:
            outcome_like_count += 1

    if outcome_like_count >= 2:
        warnings.append(
            "Một số mục tiêu cụ thể có tính chất gần với chuẩn đầu ra (CĐR). "
            "Cần rà soát để phân biệt rõ giữa mục tiêu đào tạo và CĐR."
        )

    return warnings


# ── Debug Context Builder ────────────────────────────────────────────


def build_debug_context(
    *,
    context_pack,
    queries_used: list[str] | None = None,
    fallback_used: bool = False,
) -> dict[str, Any]:
    """
    Build debug info cho Step 2.1.

    Quy tắc:
      - text_preview chỉ lấy 300-500 ký tự.
      - Không trả toàn bộ nội dung tài liệu.
      - Không trả secret/token/url tạm.
    """
    used_chunks: list[dict[str, Any]] = []
    total_chars = 0

    # Collect chunks from all context groups
    groups = [
        ("current_objective", context_pack.current_objective_contexts),
        ("direction", context_pack.direction_contexts),
        ("legal", context_pack.legal_contexts),
        ("evidence", context_pack.evidence_contexts),
        ("comparison", context_pack.comparison_contexts),
    ]

    seen: set[str] = set()
    for group_key, items in groups:
        for item in items:
            dedup_key = f"{item.ai_document_id}:{item.chunk_index}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            text = item.text or ""
            total_chars += len(text)
            # Preview: max 400 chars
            preview = text[:400].strip()
            if len(text) > 400:
                preview += "…"

            used_chunks.append({
                "document_id": item.ai_document_id,
                "document_name": item.filename,
                "document_role": item.document_role,
                "chunk_index": item.chunk_index,
                "score": round(item.score, 4),
                "text_preview": preview,
                "context_group": group_key,
            })

    # Missing roles
    missing_roles: list[str] = []
    for key, cov in context_pack.role_coverage.items():
        if cov.status in ("missing", "document_available_no_context", "failed"):
            missing_roles.append(key)

    return {
        "queries": queries_used or [],
        "used_chunks": used_chunks,
        "missing_roles": missing_roles,
        "fallback_used": fallback_used,
        "context_char_count": total_chars,
    }


# ── Deduplication helper ─────────────────────────────────────────────


def deduplicate_contexts(
    contexts: list,
) -> list:
    """
    Deduplicate contexts by (ai_document_id, chunk_index).

    Keeps the highest-score version.
    """
    seen: dict[str, Any] = {}
    for ctx in contexts:
        key = f"{ctx.ai_document_id}:{ctx.chunk_index}"
        existing = seen.get(key)
        if existing is None or ctx.score > existing.score:
            seen[key] = ctx
    return list(seen.values())
