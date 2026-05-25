"""
ObjectiveUpdateSkill — Đề xuất cập nhật mục tiêu đào tạo (R6.1B).

Nhận ObjectiveUpdateContextPack → gọi LLM → trả JSON đề xuất
cập nhật mục tiêu đào tạo.

Mục tiêu đào tạo:
  - Là định hướng cấp chương trình.
  - Mô tả người học sau tốt nghiệp có thể phát triển theo hướng nào.
  - Thể hiện năng lực nghề nghiệp/phẩm chất/vai trò xã hội ở mức tổng quát.
  - Là nền để dẫn xuất chuẩn đầu ra (CĐR sẽ làm ở R6.2).

Guards:
  - Không ghi Program / ProgramVersion / ProgramVersionRevision.
  - Không sinh CĐR ở bước này.
  - Không bịa căn cứ, vị trí việc làm, quy định pháp lý.
  - Thiếu dữ liệu → missing_information / quality_flags, không bịa.
  - Proposed objective không có evidence → confidence=low hoặc missing_evidence flag.
  - Mã PO do AI đề xuất → is_draft_code=true.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


# ── Result DTOs ──────────────────────────────────────────────────────


class ObjectiveUpdateStatus:
    GENERATED = "generated"
    NEEDS_GENERATION = "needs_generation"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    FAILED = "failed"


@dataclass
class ObjectiveUpdatePayload:
    """Full payload matching the R6.1B + R6.5 + R6.5C output JSON schema."""
    objective_update_strategy: dict[str, Any] = field(default_factory=dict)
    current_objective_analysis: list[dict[str, Any]] = field(default_factory=list)
    proposed_objectives: list[dict[str, Any]] = field(default_factory=list)
    alignment_notes: list[dict[str, Any]] = field(default_factory=list)
    objective_quality_review: dict[str, Any] = field(default_factory=dict)
    missing_information: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    next_actions: list[dict[str, Any]] = field(default_factory=list)
    # R6.5: Direct flat output for Laravel binding
    general_objective_text: str = ""
    specific_objective_texts: list[str] = field(default_factory=list)
    # R6.5C: Structured objectives with M-codes and group allocation
    general_objective: str = ""
    specific_objectives_structured: list[dict[str, Any]] = field(default_factory=list)
    objective_count: int = 6
    format_profile: str = ""
    evidence_quality: str = "moderate"


@dataclass
class ObjectiveUpdateResult:
    """Full result from objective update skill."""
    status: str
    payload: ObjectiveUpdatePayload
    warnings: list[str]
    task_type: str = "objective_update"


# ── Prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Bạn là chuyên gia tư vấn thiết kế chương trình đào tạo đại học Việt Nam, \
chuyên sâu về MỤC TIÊU ĐÀO TẠO.

NHIỆM VỤ: Dựa trên CTĐT hiện hành, chỉ đạo của nhà trường, quy định pháp lý, \
khảo sát bên liên quan, và đối sánh → VIẾT mục tiêu đào tạo cập nhật.

MỤC TIÊU ĐÀO TẠO LÀ GÌ:
- Là định hướng CẤP CHƯƠNG TRÌNH (không phải cấp học phần).
- Mô tả người học sau tốt nghiệp có thể phát triển theo HƯỚNG NÀO.
- Thể hiện năng lực nghề nghiệp, phẩm chất, vai trò xã hội ở MỨC TỔNG QUÁT.
- Là nền để dẫn xuất chuẩn đầu ra (CĐR), KHÔNG PHẢI CĐR.

YÊU CẦU VỀ VĂN PHONG VÀ NỘI DUNG:
- Viết bằng tiếng Việt học thuật, trang trọng, phù hợp văn bản CTĐT đại học.
- KHÔNG viết chung chung áp dụng cho mọi ngành — phải bám vào ngành/chuyên ngành \
phát hiện trong tài liệu.
- Nếu không xác định được ngành/chương trình, thêm missing_information thay vì tự bịa.
- KHÔNG biến mục tiêu đào tạo thành chuẩn đầu ra. KHÔNG liệt kê chi tiết như CLO/PLO.
- Nếu context có thông tin cụ thể về ngành, công nghệ, nghiệp vụ, môi trường làm việc, \
vị trí việc làm → đưa vào nội dung ở mức mục tiêu đào tạo.
- TRÁNH các câu rỗng như: "có kiến thức và kỹ năng cần thiết", \
"đáp ứng nhu cầu xã hội", "có phẩm chất đạo đức tốt" nếu không có phần cụ thể đi kèm.

YÊU CẦU VỀ CẤU TRÚC MỤC TIÊU:
- general_objective_text: MỘT ĐOẠN VĂN hoàn chỉnh, khoảng 120-180 từ, mô tả tổng quát \
mục tiêu đào tạo của chương trình. Phải đề cập ngành/lĩnh vực cụ thể.
- specific_objective_texts: DANH SÁCH 4-6 mục tiêu cụ thể, mỗi mục 1-2 câu. \
Phải bao phủ các khía cạnh sau:
  1. Kiến thức nền tảng và chuyên môn (đặc thù ngành)
  2. Kỹ năng nghề nghiệp/thực hành
  3. Năng lực ứng dụng, phân tích, thiết kế, giải quyết vấn đề
  4. Thái độ, đạo đức nghề nghiệp, trách nhiệm xã hội
  5. Khả năng tự học, nghiên cứu, thích ứng với thay đổi (nếu phù hợp)

QUY TẮC BẮT BUỘC:
1. CHỈ đề xuất dựa trên các đoạn trích (contexts) được cung cấp.
2. KHÔNG bịa thêm thông tin, số liệu, nhu cầu doanh nghiệp, hoặc nguồn \
không có trong contexts.
3. KHÔNG bịa quy định pháp lý, thông tư, nếu không có trong contexts.
4. KHÔNG sinh chuẩn đầu ra (CĐR). Đó là bước khác.
5. Nếu mã PO1/PO2 do bạn tự đề xuất, set is_draft_code=true.
6. Nếu proposed_content quá chi tiết giống CĐR, thêm quality_flags: \
["overlaps_with_outcome"].
7. Nếu proposed objective KHÔNG có evidence_refs, set confidence="low" \
VÀ thêm quality_flags: ["missing_evidence"].
8. Nếu thiếu dữ liệu, đưa vào missing_information, KHÔNG bịa.
9. Mỗi đề xuất phải gắn source_indices (chỉ số 0-indexed) trong evidence_refs.
10. objective_type phải là "general_objective" hoặc "specific_objective".

OUTPUT FORMAT (JSON):
{
  "general_objective_text": "Một đoạn văn hoàn chỉnh 120-180 từ mô tả mục tiêu chung \
của chương trình đào tạo, bám sát ngành/chuyên ngành cụ thể.",
  "specific_objective_texts": [
    "Mục tiêu cụ thể 1: Kiến thức nền tảng và chuyên môn...",
    "Mục tiêu cụ thể 2: Kỹ năng nghề nghiệp/thực hành...",
    "Mục tiêu cụ thể 3: Năng lực ứng dụng...",
    "Mục tiêu cụ thể 4: Thái độ, đạo đức nghề nghiệp..."
  ],
  "objective_update_strategy": {
    "summary": "Tóm tắt chiến lược cập nhật mục tiêu (2-3 câu)",
    "main_drivers": ["school_direction","legal_regulation","stakeholder_need",\
"labor_market","curriculum_gap","comparison_gap","other"],
    "human_review_required": true
  },
  "current_objective_analysis": [
    {
      "current_objective": "Mục tiêu hiện tại (trích từ contexts)",
      "issue": "Vấn đề cần cải thiện",
      "evidence_refs": [{"source_index": 0, "context_group": "current_objective"}],
      "confidence": "low|medium|high"
    }
  ],
  "proposed_objectives": [
    {
      "objective_type": "general_objective|specific_objective",
      "code": "PO1 hoặc null",
      "is_draft_code": true,
      "update_operation": "keep|revise|replace|add|remove",
      "mapped_from_current": "Mục tiêu hiện tại được map (nếu có)",
      "proposed_content": "Nội dung mục tiêu đề xuất",
      "rationale": "Lý do",
      "alignment": {
        "school_direction": "...",
        "legal_regulation": "...",
        "stakeholder_need": "...",
        "labor_market": "...",
        "current_curriculum_gap": "...",
        "comparison_gap": "..."
      },
      "evidence_refs": [{"source_index": 0, "context_group": "direction"}],
      "quality_flags": [],
      "priority": "low|medium|high",
      "confidence": "low|medium|high"
    }
  ],
  "alignment_notes": [
    {
      "target": "school_direction|legal_regulation|stakeholder_need|\
labor_market|current_curriculum|comparison|other",
      "note": "Ghi chú alignment",
      "evidence_refs": [{"source_index": 0, "context_group": "legal"}]
    }
  ],
  "objective_quality_review": {
    "overall_assessment": "Đánh giá tổng thể bộ mục tiêu đề xuất",
    "strengths": ["Điểm mạnh"],
    "weaknesses": ["Điểm yếu"],
    "consistency_notes": ["Ghi chú nhất quán"],
    "recommendation_for_human_review": ["Đề xuất cho hội đồng"]
  },
  "missing_information": [
    {
      "type": "current_objectives|direction_decision|legal_regulation|\
survey_evidence|comparison_report|other",
      "description": "Mô tả thông tin còn thiếu"
    }
  ],
  "risks": [
    {
      "risk": "Mô tả rủi ro",
      "impact": "low|medium|high",
      "mitigation": "Biện pháp giảm thiểu"
    }
  ],
  "next_actions": [
    {
      "action": "Hành động cần thực hiện",
      "owner_hint": "bo_mon|khoa|hoi_dong|admin|unknown",
      "priority": "low|medium|high"
    }
  ]
}

Chỉ trả JSON, không giải thích thêm.\
"""


# ── R6.5C: Objective count allocation ────────────────────────────────

_ALLOCATION_TABLE: dict[int, dict[str, list[str]]] = {
    4: {"knowledge": ["M1", "M2"], "skills_attitude": ["M3"], "foreign_language_it": ["M4"]},
    5: {"knowledge": ["M1", "M2", "M3"], "skills_attitude": ["M4"], "foreign_language_it": ["M5"]},
    6: {"knowledge": ["M1", "M2", "M3"], "skills_attitude": ["M4", "M5"], "foreign_language_it": ["M6"]},
    7: {"knowledge": ["M1", "M2", "M3"], "skills_attitude": ["M4", "M5", "M6"], "foreign_language_it": ["M7"]},
    8: {"knowledge": ["M1", "M2", "M3", "M4"], "skills_attitude": ["M5", "M6", "M7"], "foreign_language_it": ["M8"]},
}


def compute_objective_allocation(objective_count: int) -> dict[str, list[str]]:
    """Return M-code allocation per group for given objective_count (4-8)."""
    clamped = max(4, min(8, objective_count))
    return _ALLOCATION_TABLE[clamped]


def _build_structured_system_prompt(objective_count: int) -> str:
    """Build system prompt for R6.5C structured objective generation."""
    alloc = compute_objective_allocation(objective_count)

    alloc_desc_lines = []
    for group, codes in alloc.items():
        alloc_desc_lines.append(f"  - Nhóm {group}: {', '.join(codes)}")
    alloc_block = "\n".join(alloc_desc_lines)

    return f"""\
Bạn là trợ lý hỗ trợ cập nhật chương trình đào tạo đại học Việt Nam.

NHIỆM VỤ: Sinh phần "Mục tiêu đào tạo" theo văn phong CTĐT chính thức, \
bám tài liệu nguồn/RAG context đã cung cấp.

QUY TẮC BẮT BUỘC:
1. Sinh đúng {objective_count} mục tiêu cụ thể, mã từ M1 đến M{objective_count}.
2. Không sinh thiếu, không sinh dư, không nhảy số, không trùng mã.
3. Chia nhóm theo phân bổ sau:
{alloc_block}
4. Mục tiêu cuối cùng LUÔN thuộc nhóm foreign_language_it.
5. Không tự bịa thông tin nếu context không có.
6. Nếu thiếu minh chứng, vẫn sinh bản nháp dựa trên thông tin chắc chắn \
(tên ngành, chuẩn đầu ra, khối kiến thức, học phần) nhưng thêm warning.

NỘI DUNG TỪNG NHÓM:
- Nhóm knowledge (Về kiến thức): kiến thức chung, kiến thức cơ sở ngành, \
kiến thức chuyên ngành. Bám vào ngành/chương trình từ context.
- Nhóm skills_attitude (Về kỹ năng, thái độ): kỹ năng cá nhân, tự học, \
kỹ năng nghề nghiệp, giao tiếp, làm việc nhóm, tổ chức công việc, \
tư duy tích cực, sáng tạo, thích nghi, đạo đức nghề nghiệp, \
trách nhiệm xã hội, phát triển bền vững.
- Nhóm foreign_language_it (Trình độ Ngoại ngữ, Tin học): tiếng Anh, \
ngoại ngữ, tin học, năng lực số, công cụ số, phục vụ học tập, \
nghiên cứu khoa học, làm việc trong môi trường chuyên môn. \
Nếu ngành CNTT, tránh viết "tin học cơ bản" quá sơ cấp — \
viết theo hướng năng lực ngoại ngữ, công cụ số, năng lực số.

MỤC TIÊU CHUNG (general_objective):
- Một đoạn văn hoàn chỉnh, 120-180 từ.
- Bắt đầu: "Mục tiêu của chương trình là đào tạo ..."
- Có tên ngành/chương trình nếu context có.
- Đề cập: kiến thức chung, kiến thức cơ sở, kiến thức chuyên ngành, \
kỹ năng thực hành, làm việc độc lập, sáng tạo, phân tích, giải quyết vấn đề, \
đạo đức, trách nhiệm, sức khỏe, năng lực học tập và nghiên cứu suốt đời.
- Không viết kiểu marketing. Không quá chung chung.
- Viết tiếng Việt học thuật, trang trọng.

OUTPUT FORMAT (chỉ JSON, không markdown, không giải thích):
{{
  "general_objective": "Mục tiêu của chương trình là đào tạo ...",
  "specific_objectives": [
    {{"code": "M1", "group": "knowledge", "text": "..."}},
    {{"code": "M2", "group": "knowledge", "text": "..."}},
    ...
    {{"code": "M{objective_count}", "group": "foreign_language_it", "text": "..."}}
  ],
  "evidence_quality": "strong|moderate|weak",
  "warnings": []
}}

Chỉ trả JSON, không giải thích thêm.\
"""


# R6.5: Variable context char limits per role group
_CONTEXT_CHAR_LIMITS: dict[str, int] = {
    "current_objective": 2500,
    "direction": 1200,
    "evidence": 1200,
    "comparison": 1200,
    "legal": 900,
}
_DEFAULT_CHAR_LIMIT = 800
# Safety net: total context chars to avoid prompt overflow
_MAX_TOTAL_CONTEXT_CHARS = 14000


def _build_user_prompt(
    *,
    program_name: str | None,
    program_code: str | None,
    update_cycle_id: str,
    context_pack,
    user_instruction: str | None = None,
    objective_count: int | None = None,
) -> tuple[str, dict[str, list]]:
    """Build user prompt from context pack. Returns (prompt, source_map)."""
    header = f"Đợt cập nhật CTĐT: {update_cycle_id}"
    if program_name:
        header += f"\nChương trình: {program_name}"
    if program_code:
        header += f" (Mã ngành: {program_code})"

    if user_instruction:
        header += f"\n\nHướng dẫn bổ sung: {user_instruction}"

    # Build contexts by group with global indexing
    source_map: dict[str, list] = {}
    all_parts: list[str] = []
    global_idx = 0
    total_chars = 0

    groups = [
        ("current_objective", context_pack.current_objective_contexts,
         "CTĐT HIỆN HÀNH — mục tiêu đào tạo"),
        ("direction", context_pack.direction_contexts,
         "CHỈ ĐẠO / QUYẾT ĐỊNH CẬP NHẬT"),
        ("legal", context_pack.legal_contexts,
         "QUY ĐỊNH PHÁP LÝ"),
        ("evidence", context_pack.evidence_contexts,
         "KHẢO SÁT / BIÊN BẢN HỌP"),
        ("comparison", context_pack.comparison_contexts,
         "ĐỐI SÁNH CTĐT"),
    ]

    for group_key, items, label in groups:
        if not items:
            all_parts.append(f"\n=== {label} ===\n(Không có dữ liệu)")
            continue

        # R6.5: Variable char limit per group
        char_limit = _CONTEXT_CHAR_LIMITS.get(group_key, _DEFAULT_CHAR_LIMIT)

        all_parts.append(f"\n=== {label} ===")
        group_sources = []
        for item in items:
            # R6.5: Apply per-group char limit and total safety net
            remaining = _MAX_TOTAL_CONTEXT_CHARS - total_chars
            if remaining <= 0:
                break
            effective_limit = min(char_limit, remaining)
            text = item.text[:effective_limit] if item.text else "(trống)"
            total_chars += len(text)

            fname = item.filename or "N/A"
            role = item.document_role or "unknown"
            all_parts.append(
                f"[Context {global_idx}] group={group_key} file={fname} "
                f"role={role} score={item.score:.3f}\n{text}"
            )
            group_sources.append({"global_index": global_idx, "item": item})
            global_idx += 1
        source_map[group_key] = group_sources

    # Missing info from context pack
    if context_pack.missing_information:
        mi_lines = [f"- {m['type']}: {m['description']}" for m in context_pack.missing_information]
        all_parts.append(
            "\n=== THÔNG TIN THIẾU ===\n" + "\n".join(mi_lines)
        )

    contexts_block = "\n\n".join(all_parts) if all_parts else "(Không có contexts)"
    prompt = f"{header}\n\n{contexts_block}"
    return prompt, source_map


# ── Overlap-with-outcome validator ───────────────────────────────

# Action verbs that are characteristic of CĐR (chuẩn đầu ra)
# rather than mục tiêu đào tạo (which is higher-level/strategic)
_OUTCOME_VERBS = frozenset([
    "phân tích", "thiết kế", "triển khai", "đánh giá",
    "vận dụng", "xây dựng", "kiểm thử",
])

_OVERLAP_THRESHOLD = 3  # ≥3 action verbs → likely an outcome, not an objective


def _looks_like_outcome(proposed_content: str, objective_type: str) -> bool:
    """
    Lightweight heuristic: if a *general_objective* contains too many
    specific action verbs, it probably overlaps with CĐR.

    Does NOT block output — only flags for human review.
    """
    if not proposed_content:
        return False
    content_lower = proposed_content.lower()
    hit_count = sum(1 for v in _OUTCOME_VERBS if v in content_lower)
    # general_objective with many action verbs is suspect
    if objective_type == "general_objective" and hit_count >= _OVERLAP_THRESHOLD:
        return True
    # any type with very high verb density
    if hit_count >= _OVERLAP_THRESHOLD + 1:
        return True
    return False


# ── Skill ────────────────────────────────────────────────────────────


class ObjectiveUpdateSkill:
    """
    AI skill for proposing objective updates.

    Fail-open design: LLM errors produce status=failed with warnings.

    Guards:
      - No writes to Program/ProgramVersion/ProgramVersionRevision.
      - No CĐR generation (that's R6.2).
      - Missing data → missing_information / quality_flags, never fabrication.
    """

    async def run(
        self,
        *,
        update_cycle_id: str,
        program_id: str | None = None,
        program_code: str | None = None,
        program_name: str | None = None,
        context_pack,
        user_instruction: str | None = None,
        objective_count: int = 6,
    ) -> ObjectiveUpdateResult:
        """Run objective update skill on the provided context pack."""
        # ── Count total contexts ─────────────────────────────────
        total = (
            len(context_pack.current_objective_contexts)
            + len(context_pack.direction_contexts)
            + len(context_pack.legal_contexts)
            + len(context_pack.evidence_contexts)
            + len(context_pack.comparison_contexts)
        )

        # ── No contexts → insufficient_context ───────────────────
        if total == 0:
            return ObjectiveUpdateResult(
                status=ObjectiveUpdateStatus.INSUFFICIENT_CONTEXT,
                payload=ObjectiveUpdatePayload(
                    objective_update_strategy={
                        "summary": "",
                        "main_drivers": [],
                        "human_review_required": True,
                    },
                    objective_quality_review={
                        "overall_assessment": "Không đủ dữ liệu để đề xuất cập nhật mục tiêu đào tạo.",
                        "strengths": [],
                        "weaknesses": ["Thiếu tài liệu đầu vào."],
                        "consistency_notes": [],
                        "recommendation_for_human_review": [
                            "Cần bổ sung tài liệu CTĐT hiện hành và chỉ đạo cập nhật."
                        ],
                    },
                    missing_information=list(context_pack.missing_information),
                ),
                warnings=["Không có contexts nào trong context pack."],
            )

        # ── LLM disabled → deterministic fallback ────────────────
        if not getattr(settings, "SYNTHESIS_ENABLED", False):
            return ObjectiveUpdateResult(
                status=ObjectiveUpdateStatus.NEEDS_GENERATION,
                payload=ObjectiveUpdatePayload(
                    missing_information=list(context_pack.missing_information),
                ),
                warnings=[
                    "SYNTHESIS_ENABLED=false. Cần bật LLM để sinh đề xuất mục tiêu.",
                    f"Có {total} contexts sẵn sàng.",
                ],
            )

        # ── Check OpenAI key ─────────────────────────────────────
        api_key = getattr(settings, "OPENAI_API_KEY", None)
        if not api_key:
            return ObjectiveUpdateResult(
                status=ObjectiveUpdateStatus.NEEDS_GENERATION,
                payload=ObjectiveUpdatePayload(
                    missing_information=list(context_pack.missing_information),
                ),
                warnings=[
                    "OPENAI_API_KEY chưa được cấu hình.",
                    f"Có {total} contexts sẵn sàng.",
                ],
            )

        # ── Build prompt ─────────────────────────────────────────
        # R6.5C: clamp objective_count
        objective_count = max(4, min(8, objective_count))

        user_prompt, source_map = _build_user_prompt(
            program_name=program_name,
            program_code=program_code,
            update_cycle_id=update_cycle_id,
            context_pack=context_pack,
            user_instruction=user_instruction,
            objective_count=objective_count,
        )

        # R6.5C: Use structured prompt for dynamic M-code generation
        system_prompt = _build_structured_system_prompt(objective_count)

        # ── Call LLM ─────────────────────────────────────────────
        try:
            raw_json = await self._call_openai(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                api_key=api_key,
            )
        except Exception as exc:
            logger.warning(
                "objective_update_skill.llm_failed update_cycle=%s error=%s",
                update_cycle_id, exc.__class__.__name__,
            )
            return ObjectiveUpdateResult(
                status=ObjectiveUpdateStatus.FAILED,
                payload=ObjectiveUpdatePayload(
                    missing_information=list(context_pack.missing_information),
                ),
                warnings=[
                    f"LLM gọi thất bại: {exc.__class__.__name__}",
                    f"Có {total} contexts nhưng chưa thể phân tích.",
                ],
            )

        # ── Parse response ───────────────────────────────────────
        try:
            result = self._parse_response(
                raw_json, source_map, context_pack,
                objective_count=objective_count,
            )
        except Exception as exc:
            logger.warning(
                "objective_update_skill.parse_failed update_cycle=%s error=%s",
                update_cycle_id, str(exc)[:200],
            )
            return ObjectiveUpdateResult(
                status=ObjectiveUpdateStatus.FAILED,
                payload=ObjectiveUpdatePayload(
                    missing_information=list(context_pack.missing_information),
                ),
                warnings=[f"Lỗi parse LLM response: {str(exc)[:150]}"],
            )

        logger.info(
            "objective_update_skill.done update_cycle=%s "
            "proposed=%d status=%s",
            update_cycle_id,
            len(result.payload.proposed_objectives),
            result.status,
        )
        return result

    # ── OpenAI call ──────────────────────────────────────────────

    async def _call_openai(
        self, *, system_prompt: str, user_prompt: str, api_key: str,
    ) -> str:
        model = getattr(settings, "SYNTHESIS_MODEL", "gpt-4o-mini")
        timeout_s = float(getattr(settings, "SYNTHESIS_TIMEOUT_S", 90.0))
        max_tokens = int(getattr(settings, "SYNTHESIS_MAX_TOKENS", 8192))
        temperature = float(getattr(settings, "SYNTHESIS_TEMPERATURE", 0.15))

        payload = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=timeout_s + 2.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty LLM response")
        return text.strip()

    # ── Parse + validate ─────────────────────────────────────────

    def _parse_response(
        self, raw_json: str, source_map: dict[str, list], context_pack,
        *, objective_count: int = 6,
    ) -> ObjectiveUpdateResult:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")

        warnings: list[str] = []

        def _resolve_refs(raw_refs: list | None) -> list[dict[str, Any]]:
            if not raw_refs or not isinstance(raw_refs, list):
                return []
            resolved = []
            for ref in raw_refs:
                if not isinstance(ref, dict):
                    continue
                idx = ref.get("source_index")
                group = ref.get("context_group", "")
                sources = source_map.get(group, [])
                matched = None
                for s in sources:
                    if s["global_index"] == idx:
                        matched = s["item"]
                        break
                if matched is None:
                    # Try flat index across all groups
                    all_items = []
                    for g in source_map.values():
                        all_items.extend(g)
                    for s in all_items:
                        if s["global_index"] == idx:
                            matched = s["item"]
                            break
                if matched is not None:
                    aid = getattr(matched, "ai_document_id", None)
                    if aid is None:
                        continue
                    source_meta = getattr(matched, "source", None) or {}
                    resolved.append({
                        "ai_document_id": aid,
                        "external_file_id": getattr(matched, "external_file_id", None),
                        "filename": getattr(matched, "filename", None),
                        "document_role": getattr(matched, "document_role", None),
                        "chunk_id": getattr(matched, "chunk_id", None),
                        "chunk_index": getattr(matched, "chunk_index", None),
                        "score": getattr(matched, "score", 0.0),
                        "quote": (getattr(matched, "text", "") or "")[:200],
                        "update_cycle_id": source_meta.get("update_cycle_id"),
                        "program_code": source_meta.get("program_code"),
                        "program_id": source_meta.get("program_id"),
                    })
                elif isinstance(idx, int):
                    warnings.append(f"source_index {idx} not resolved")
            return resolved

        def _safe_str(val: Any, default: str = "") -> str:
            return str(val).strip() if val else default

        # ── Parse sections ───────────────────────────────────────
        strategy = data.get("objective_update_strategy", {})
        if not isinstance(strategy, dict):
            strategy = {}

        current_analysis = []
        for raw in data.get("current_objective_analysis", []):
            if not isinstance(raw, dict):
                continue
            current_analysis.append({
                "current_objective": _safe_str(raw.get("current_objective")),
                "issue": _safe_str(raw.get("issue")),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
                "confidence": _safe_str(raw.get("confidence"), "medium"),
            })

        proposed = []
        for raw in data.get("proposed_objectives", []):
            if not isinstance(raw, dict):
                continue
            refs = _resolve_refs(raw.get("evidence_refs"))
            flags = list(raw.get("quality_flags", []))
            confidence = _safe_str(raw.get("confidence"), "medium")

            # Business rule: no evidence → low confidence + missing_evidence
            if not refs:
                confidence = "low"
                if "missing_evidence" not in flags:
                    flags.append("missing_evidence")

            # Business rule: draft codes
            code = raw.get("code")
            is_draft = raw.get("is_draft_code", True)
            if code is not None:
                is_draft = True  # AI-proposed codes are always draft

            alignment = raw.get("alignment", {})
            if not isinstance(alignment, dict):
                alignment = {}

            proposed_content = _safe_str(raw.get("proposed_content"))
            obj_type = _safe_str(
                raw.get("objective_type"), "specific_objective"
            )

            # Post-parse validator: detect CĐR-like content
            if _looks_like_outcome(proposed_content, obj_type):
                if "overlaps_with_outcome" not in flags:
                    flags.append("overlaps_with_outcome")
                if "needs_human_review" not in flags:
                    flags.append("needs_human_review")

            proposed.append({
                "objective_type": obj_type,
                "code": code,
                "is_draft_code": is_draft,
                "update_operation": _safe_str(
                    raw.get("update_operation"), "add"
                ),
                "mapped_from_current": _safe_str(
                    raw.get("mapped_from_current")
                ),
                "proposed_content": proposed_content,
                "rationale": _safe_str(raw.get("rationale")),
                "alignment": alignment,
                "evidence_refs": refs,
                "quality_flags": flags,
                "priority": _safe_str(raw.get("priority"), "medium"),
                "confidence": confidence,
            })

        alignment_notes = []
        for raw in data.get("alignment_notes", []):
            if not isinstance(raw, dict):
                continue
            alignment_notes.append({
                "target": _safe_str(raw.get("target"), "other"),
                "note": _safe_str(raw.get("note")),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
            })

        quality_review = data.get("objective_quality_review", {})
        if not isinstance(quality_review, dict):
            quality_review = {}

        missing_info = []
        for raw in data.get("missing_information", []):
            if not isinstance(raw, dict):
                continue
            missing_info.append({
                "type": _safe_str(raw.get("type"), "other"),
                "description": _safe_str(raw.get("description")),
            })

        risks = []
        for raw in data.get("risks", []):
            if not isinstance(raw, dict):
                continue
            risks.append({
                "risk": _safe_str(raw.get("risk")),
                "impact": _safe_str(raw.get("impact"), "medium"),
                "mitigation": _safe_str(raw.get("mitigation")),
            })

        next_actions = []
        for raw in data.get("next_actions", []):
            if not isinstance(raw, dict):
                continue
            next_actions.append({
                "action": _safe_str(raw.get("action")),
                "owner_hint": _safe_str(raw.get("owner_hint"), "unknown"),
                "priority": _safe_str(raw.get("priority"), "medium"),
            })

        # R6.5: Extract direct flat output fields
        general_text = _safe_str(data.get("general_objective_text"))
        specific_texts_raw = data.get("specific_objective_texts")
        specific_texts: list[str] = []
        if isinstance(specific_texts_raw, list):
            specific_texts = [
                _safe_str(s) for s in specific_texts_raw
                if isinstance(s, str) and s.strip()
            ]

        # R6.5C: Parse structured objectives from new prompt format
        general_objective = _safe_str(data.get("general_objective"))
        structured_objectives: list[dict[str, Any]] = []
        structured_texts: list[str] = []

        raw_specific = data.get("specific_objectives")
        has_structured_contract_response = isinstance(raw_specific, list)
        if isinstance(raw_specific, list):
            for item in raw_specific:
                if not isinstance(item, dict):
                    continue
                code = _safe_str(item.get("code"))
                group = _safe_str(item.get("group"))
                text = _safe_str(item.get("text"))
                if code or group or text:
                    structured_objectives.append({
                        "code": code,
                        "group": group,
                        "text": text,
                    })
                    if code and text:
                        structured_texts.append(f"{code}. {text}")

        # R6.9A: Strict M-code count and group contract.
        expected_alloc = compute_objective_allocation(objective_count)
        expected_codes = [f"M{i}" for i in range(1, objective_count + 1)]
        expected_group_by_code = {
            code: group
            for group, codes in expected_alloc.items()
            for code in codes
        }
        actual_codes = [o["code"] for o in structured_objectives]
        contract_warnings: list[str] = []

        if not has_structured_contract_response:
            contract_warnings.append(
                f"AI không trả về cấu trúc mục tiêu bắt buộc M1 đến M{objective_count}."
            )
        else:
            actual_objective_count = len(structured_objectives)
            if actual_objective_count < objective_count:
                contract_warnings.append(
                    f"AI chỉ sinh được {actual_objective_count}/{objective_count} "
                    "mục tiêu cụ thể hợp lệ."
                )
            elif actual_objective_count > objective_count:
                contract_warnings.append(
                    f"AI sinh {actual_objective_count}/{objective_count} mục tiêu cụ thể, "
                    "vượt số lượng được yêu cầu."
                )

            if actual_codes != expected_codes:
                contract_warnings.append(
                    f"Mã mục tiêu không đúng yêu cầu. Kỳ vọng M1 đến M{objective_count}."
                )

            duplicate_codes = sorted({
                code for code in actual_codes
                if code and actual_codes.count(code) > 1
            })
            if duplicate_codes:
                contract_warnings.append(
                    "Mã mục tiêu bị trùng: " + ", ".join(duplicate_codes) + "."
                )

            for obj in structured_objectives:
                code = obj.get("code", "")
                expected_group = expected_group_by_code.get(code)
                if expected_group and obj.get("group") != expected_group:
                    contract_warnings.append(
                        f"{code} phải thuộc nhóm {expected_group} "
                        f"nhưng AI trả về {obj.get('group') or 'rỗng'}."
                    )
                if not str(obj.get("text") or "").strip():
                    contract_warnings.append(
                        f"{code or 'Một mục tiêu'} có nội dung rỗng."
                    )

        for warning in contract_warnings:
            if warning not in warnings:
                warnings.append(warning)

        # R6.5C: Use structured data to also populate legacy fields
        if general_objective and not general_text:
            general_text = general_objective
        if structured_texts and not specific_texts:
            specific_texts = structured_texts
        if general_text and not general_objective:
            general_objective = general_text

        # R6.5C: Evidence quality from LLM
        evidence_quality = _safe_str(data.get("evidence_quality"), "moderate")
        llm_warnings = data.get("warnings")
        if isinstance(llm_warnings, list):
            for w in llm_warnings:
                if isinstance(w, str) and w.strip():
                    warnings.append(w.strip())

        # R6.5C-HARDEN-1: Default warning for weak evidence
        _WEAK_EVIDENCE_DEFAULT = (
            "Nguồn minh chứng chưa đủ mạnh, "
            "nội dung cần được rà soát thủ công."
        )
        if evidence_quality == "weak":
            # Only add if no similar warning exists
            has_similar = any(
                "minh chứng" in w and "rà soát" in w
                for w in warnings
            )
            if not has_similar:
                warnings.append(_WEAK_EVIDENCE_DEFAULT)

        format_profile = (
            "tay_nguyen_ctdt_dynamic_m_objectives"
            if structured_objectives
            else ""
        )

        payload = ObjectiveUpdatePayload(
            objective_update_strategy=strategy,
            current_objective_analysis=current_analysis,
            proposed_objectives=proposed,
            alignment_notes=alignment_notes,
            objective_quality_review=quality_review,
            missing_information=missing_info,
            risks=risks,
            next_actions=next_actions,
            general_objective_text=general_text,
            specific_objective_texts=specific_texts,
            # R6.5C new fields
            general_objective=general_objective,
            specific_objectives_structured=structured_objectives,
            objective_count=objective_count,
            format_profile=format_profile,
            evidence_quality=evidence_quality,
        )

        # R6.5C: Fixed status determination — check ALL content sources
        has_content = (
            proposed
            or current_analysis
            or general_text
            or general_objective
            or specific_texts
            or structured_objectives
        )
        if contract_warnings:
            status = ObjectiveUpdateStatus.FAILED
        else:
            status = (
                ObjectiveUpdateStatus.GENERATED
                if has_content
                else ObjectiveUpdateStatus.INSUFFICIENT_CONTEXT
            )

        return ObjectiveUpdateResult(
            status=status, payload=payload, warnings=warnings,
        )
