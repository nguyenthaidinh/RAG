"""
OutcomeUpdateSkill — Đề xuất cập nhật chuẩn đầu ra/CĐR/PLO (R6.2B).

Nhận OutcomeUpdateContextPack → gọi LLM → trả JSON đề xuất CĐR.

Guards:
  - Không ghi Program / ProgramVersion / ProgramVersionRevision.
  - Không sinh học phần/ma trận ở bước này.
  - Không bịa căn cứ pháp lý, chuẩn nghề nghiệp, vị trí việc làm.
  - Thiếu dữ liệu → missing_information / quality_flags, không bịa.
  - Proposed outcome không có evidence → confidence=low hoặc missing_evidence flag.
  - Mã PLO/CĐR do AI đề xuất → is_draft_code=true.
  - CĐR phải dẫn xuất từ mục tiêu đào tạo nếu có objective_update_payload.
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


class OutcomeUpdateStatus:
    GENERATED = "generated"
    NEEDS_GENERATION = "needs_generation"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    FAILED = "failed"


@dataclass
class OutcomeUpdatePayload:
    """Full payload matching the R6.2B output JSON schema."""
    outcome_update_strategy: dict[str, Any] = field(default_factory=dict)
    current_outcome_analysis: list[dict[str, Any]] = field(default_factory=list)
    proposed_outcomes: list[dict[str, Any]] = field(default_factory=list)
    objective_outcome_alignment: list[dict[str, Any]] = field(default_factory=list)
    outcome_quality_review: dict[str, Any] = field(default_factory=dict)
    missing_information: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    next_actions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OutcomeUpdateResult:
    """Full result from outcome update skill."""
    status: str
    payload: OutcomeUpdatePayload
    warnings: list[str]
    task_type: str = "outcome_update"


# ── Validators ───────────────────────────────────────────────────────

_BROAD_PHRASES = frozenset([
    "đào tạo nguồn nhân lực", "có phẩm chất tốt", "phát triển toàn diện",
    "trở thành công dân", "đáp ứng nhu cầu xã hội", "có đạo đức nghề nghiệp",
])

_COURSE_SPECIFIC_SIGNALS = frozenset([
    "visual studio", "android studio", "matlab", "autocad", "solidworks",
    "chương ", "bài ", "tuần ", "tiết ", "buổi thực hành",
])


def _looks_too_broad(content: str) -> bool:
    if not content:
        return False
    cl = content.lower()
    hits = sum(1 for p in _BROAD_PHRASES if p in cl)
    return hits >= 2


def _looks_too_course_specific(content: str) -> bool:
    if not content:
        return False
    cl = content.lower()
    return any(s in cl for s in _COURSE_SPECIFIC_SIGNALS)


# ── Enum normalizers ─────────────────────────────────────────────────

_ALLOWED_OUTCOME_TYPE = frozenset(["knowledge", "skills", "autonomy_responsibility", "other"])
_ALLOWED_BLOOM = frozenset(["remember", "understand", "apply", "analyze", "evaluate", "create", "unknown"])
_ALLOWED_UPDATE_OP = frozenset(["keep", "revise", "replace", "add", "remove"])
_ALLOWED_PRIORITY = frozenset(["low", "medium", "high"])
_ALLOWED_CONFIDENCE = frozenset(["low", "medium", "high"])
_ALLOWED_COVERAGE = frozenset(["covered", "partially_covered", "not_covered", "unknown"])
_ALLOWED_QUALITY_FLAGS = frozenset([
    "too_broad", "too_specific", "missing_evidence", "overlaps_with_objective",
    "too_course_specific", "missing_objective_mapping", "needs_human_review",
])


def _norm_enum(value: str, allowed: frozenset, default: str, field_name: str, warnings: list[str]) -> str:
    """Normalize an enum value. Append warning if changed."""
    v = str(value).strip().lower() if value else default
    if v in allowed:
        return v
    warnings.append(f"Normalized invalid {field_name} '{value}' to '{default}'.")
    return default


def _norm_quality_flags(raw_flags: list, warnings: list[str]) -> list[str]:
    """Deduplicate, remove invalid flags, keep stable order."""
    seen: set[str] = set()
    result: list[str] = []
    for f in raw_flags:
        fs = str(f).strip().lower() if f else ""
        if not fs or fs in seen:
            continue
        if fs in _ALLOWED_QUALITY_FLAGS:
            seen.add(fs)
            result.append(fs)
        else:
            warnings.append(f"Removed invalid quality_flag '{f}', added 'needs_human_review'.")
            if "needs_human_review" not in seen:
                seen.add("needs_human_review")
                result.append("needs_human_review")
    return result


# ── Outcome count & allocation helpers ───────────────────────────────

_GROUP_ORDER = ("knowledge", "skills", "autonomy_responsibility")
_VALID_GROUPS = frozenset(_GROUP_ORDER)

# Fixed allocation table — must match Laravel CycleOutcomeDraftService::ALLOCATIONS
_OUTCOME_ALLOCATION_TABLE: dict[int, dict[str, int]] = {
    6:  {"knowledge": 2, "skills": 3, "autonomy_responsibility": 1},
    7:  {"knowledge": 3, "skills": 3, "autonomy_responsibility": 1},
    8:  {"knowledge": 3, "skills": 4, "autonomy_responsibility": 1},
    9:  {"knowledge": 3, "skills": 4, "autonomy_responsibility": 2},
    10: {"knowledge": 4, "skills": 4, "autonomy_responsibility": 2},
    11: {"knowledge": 4, "skills": 5, "autonomy_responsibility": 2},
    12: {"knowledge": 4, "skills": 6, "autonomy_responsibility": 2},
    13: {"knowledge": 5, "skills": 6, "autonomy_responsibility": 2},
    14: {"knowledge": 5, "skills": 7, "autonomy_responsibility": 2},
    15: {"knowledge": 6, "skills": 7, "autonomy_responsibility": 2},
}


def _compute_default_allocation(outcome_count: int) -> dict[str, int]:
    """Return fixed allocation from table. Defensive clamp to 6..15."""
    clamped = max(6, min(15, outcome_count))
    return dict(_OUTCOME_ALLOCATION_TABLE[clamped])


def _is_empty_outcome(po: dict) -> bool:
    """Check if an outcome item has no real content."""
    content = (po.get("proposed_content") or "").strip()
    return not content or content.startswith("[")


def _postprocess_outcomes(
    proposed: list[dict],
    outcome_count: int,
    group_allocation: dict[str, int],
    warnings: list[str],
) -> list[dict]:
    """Post-process: filter empties, assign C-codes by position, normalize groups.

    FIX 2: No placeholder padding — only real AI content is kept.
    FIX 3: Normalize group by position order, preserving all real content.
    """
    # Filter out empty/invalid items
    valid = [po for po in proposed if not _is_empty_outcome(po)]
    if len(valid) < len(proposed):
        dropped = len(proposed) - len(valid)
        warnings.append(f"Loại bỏ {dropped} CĐR rỗng/không hợp lệ từ AI output.")

    # Truncate if LLM produced more than requested
    if len(valid) > outcome_count:
        warnings.append(
            f"AI sinh {len(valid)} CĐR, chỉ giữ {outcome_count} item đầu tiên."
        )
        valid = valid[:outcome_count]

    # Build position→group mapping from allocation
    position_groups: list[str] = []
    for group in _GROUP_ORDER:
        count = group_allocation.get(group, 0)
        position_groups.extend([group] * count)

    # Assign C-codes and groups by position
    had_code_normalization = False
    had_group_normalization = False
    result: list[dict] = []
    for idx, item in enumerate(valid):
        expected_code = f"C{idx + 1}"
        original_code = str(item.get("code") or "").strip()
        if original_code != expected_code:
            had_code_normalization = True
        item["code"] = expected_code
        # Preserve is_draft_code=False for revise/keep (official codes)
        op = item.get("update_operation", "add")
        if op in ("revise", "keep") and item.get("mapped_from_current"):
            item.setdefault("is_draft_code", False)
        else:
            item["is_draft_code"] = True
        # Assign group by position
        if idx < len(position_groups):
            expected_group = position_groups[idx]
            if item.get("outcome_type") != expected_group:
                had_group_normalization = True
            item["outcome_type"] = expected_group
        result.append(item)

    if had_code_normalization:
        warnings.append(
            "Đã chuẩn hóa mã của một hoặc nhiều CĐR về dãy C1..Cn theo cấu trúc CTĐT bắt buộc."
        )

    if had_group_normalization:
        warnings.append(
            "Đã chuẩn hóa nhóm của một hoặc nhiều CĐR theo phân bổ CTĐT bắt buộc."
        )

    return result


# ── Prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Bạn là chuyên gia tư vấn thiết kế chương trình đào tạo đại học Việt Nam, \
chuyên sâu về CHUẨN ĐẦU RA (CĐR/PLO).

NHIỆM VỤ: Dựa trên mục tiêu đào tạo đã đề xuất, CTĐT hiện hành, quy định \
pháp lý, khảo sát, đối sánh, đề cương học phần → ĐỀ XUẤT cập nhật chuẩn đầu ra.

CHUẨN ĐẦU RA LÀ GÌ:
- Là NĂNG LỰC người học cần đạt khi tốt nghiệp.
- Cụ thể hơn mục tiêu đào tạo.
- Có thể kiểm chứng/đánh giá được.
- Phải dẫn xuất từ mục tiêu đào tạo.
- Phân loại: knowledge, skills, autonomy_responsibility, other.

QUY TẮC BẮT BUỘC:
1. CHỈ đề xuất dựa trên contexts được cung cấp.
2. KHÔNG bịa thông tin, số liệu, nhu cầu doanh nghiệp, nguồn không có.
3. KHÔNG bịa quy định pháp lý, chuẩn nghề nghiệp, vị trí việc làm.
4. KHÔNG sinh ma trận CĐR-học phần. Đó là bước khác.
5. Dùng mã CĐR theo format C1, C2, C3, ..., C{n} (n = số lượng yêu cầu). \
Luôn set is_draft_code=true.
6. Nếu CĐR quá chung giống mục tiêu, thêm quality_flags: \
["overlaps_with_objective","too_broad"].
7. Nếu CĐR quá chi tiết giống nội dung học phần, thêm quality_flags: \
["too_course_specific","needs_human_review"].
8. Nếu proposed outcome KHÔNG có evidence_refs, set confidence="low" \
VÀ thêm quality_flags: ["missing_evidence"].
9. Nếu proposed outcome không map về mục tiêu đào tạo, thêm \
quality_flags: ["missing_objective_mapping"].
10. Nếu thiếu dữ liệu, đưa vào missing_information, KHÔNG bịa.
11. bloom_level phải là remember|understand|apply|analyze|evaluate|create|unknown.
12. outcome_type phải là knowledge|skills|autonomy_responsibility|other.
13. Sinh ĐÚNG số lượng CĐR theo yêu cầu trong phần PHÂN BỔ CĐR. KHÔNG thiếu, KHÔNG dư.
14. Mã CĐR phải liên tục: C1, C2, ..., C{n}. KHÔNG nhảy mã, KHÔNG trùng mã.
15. Phân bổ nhóm: knowledge trước (C1..), skills giữa, autonomy_responsibility cuối.
16. Tuân thủ CHÍNH XÁC số lượng mỗi nhóm theo phân bổ trong user prompt.

OUTPUT FORMAT (JSON):
{
  "outcome_update_strategy": {
    "summary": "Tóm tắt chiến lược (2-3 câu)",
    "main_drivers": ["objective_update","legal_regulation","stakeholder_need",\
"labor_market","current_outcome_gap","comparison_gap","course_syllabus_gap","other"],
    "human_review_required": true
  },
  "current_outcome_analysis": [
    {
      "current_outcome": "CĐR hiện tại (trích từ contexts)",
      "issue": "Vấn đề cần cải thiện",
      "mapped_objectives": [],
      "evidence_refs": [{"source_index": 0, "context_group": "current_outcome"}],
      "confidence": "low|medium|high"
    }
  ],
  "proposed_outcomes": [
    {
      "outcome_type": "knowledge|skills|autonomy_responsibility|other",
      "code": "C1",
      "is_draft_code": true,
      "update_operation": "keep|revise|replace|add|remove",
      "mapped_from_current": "CĐR hiện tại (nếu có)",
      "proposed_content": "Nội dung CĐR đề xuất",
      "rationale": "Lý do",
      "bloom_level": "remember|understand|apply|analyze|evaluate|create|unknown",
      "mapped_objectives": [
        {"objective_code": "M1", "objective_content": "...", "mapping_reason": "..."}
      ],
      "alignment": {},
      "evidence_refs": [{"source_index": 0, "context_group": "direction"}],
      "quality_flags": [],
      "priority": "low|medium|high",
      "confidence": "low|medium|high"
    }
  ],
  "objective_outcome_alignment": [
    {
      "objective_code": "M1",
      "objective_content": "...",
      "mapped_outcomes": ["C1","C2"],
      "coverage_status": "covered|partially_covered|not_covered|unknown",
      "notes": "...",
      "evidence_refs": []
    }
  ],
  "outcome_quality_review": {
    "overall_assessment": "...",
    "strengths": [],
    "weaknesses": [],
    "consistency_notes": [],
    "recommendation_for_human_review": []
  },
  "missing_information": [],
  "risks": [],
  "next_actions": []
}

Chỉ trả JSON, không giải thích thêm.\
"""


def _objective_prompt_heading(objective_source: str) -> str:
    if objective_source == "laravel_approved_objectives":
        return "=== MỤC TIÊU ĐÀO TẠO ĐÃ DUYỆT TỪ HỆ THỐNG CTĐT ==="
    if objective_source == "legacy_laravel_approved_objectives":
        return "=== MỤC TIÊU ĐÀO TẠO TỪ HỆ THỐNG CTĐT - ĐỊNH DẠNG LEGACY, CẦN RÀ SOÁT ==="
    if objective_source == "rag_latest_objective_draft_fallback":
        return "=== BẢN NHÁP AI MỤC TIÊU ĐÀO TẠO THAM KHẢO - CHƯA PHẢI NỘI DUNG ĐÃ DUYỆT ==="
    return "=== MỤC TIÊU ĐÀO TẠO ==="


def _build_user_prompt(
    *,
    program_name: str | None,
    program_code: str | None,
    update_cycle_id: str,
    context_pack,
    user_instruction: str | None = None,
    outcome_count: int = 10,
    group_allocation: dict[str, int] | None = None,
) -> tuple[str, dict[str, list]]:
    """Build user prompt from context pack. Returns (prompt, source_map)."""
    header = f"Đợt cập nhật CTĐT: {update_cycle_id}"
    if program_name:
        header += f"\nChương trình: {program_name}"
    if program_code:
        header += f" (Mã ngành: {program_code})"
    # R6.8A: allocation block
    allocation = group_allocation or _compute_default_allocation(outcome_count)
    header += f"\n\n=== PHÂN BỔ CĐR ==="
    header += f"\nSố lượng CĐR cần sinh: {outcome_count}"
    c = 1
    for g in _GROUP_ORDER:
        n = allocation.get(g, 0)
        if n > 0:
            header += f"\n  - {g}: {n} CĐR (C{c}..C{c + n - 1})"
            c += n

    if user_instruction:
        header += f"\n\nHướng dẫn bổ sung: {user_instruction}"

    # Objective update payload
    obj_parts: list[str] = []
    objective_source = getattr(context_pack, "objective_source", "none")
    objective_heading = _objective_prompt_heading(objective_source)
    if context_pack.objective_update_payload:
        general_obj = context_pack.objective_update_payload.get("_general_objective", "")
        proposed = context_pack.objective_update_payload.get("proposed_objectives", [])
        if proposed:
            obj_lines = []
            if general_obj:
                obj_lines.append(f"  Mục tiêu chung: {general_obj}")
            for po in proposed:
                code = po.get("code", "")
                content = po.get("proposed_content", "")
                obj_lines.append(f"  - {code}: {content}")
            obj_parts.append(
                f"\n{objective_heading}\n" + "\n".join(obj_lines)
            )
    if not obj_parts:
        obj_parts.append("\n=== MỤC TIÊU ĐÀO TẠO ===\n(Chưa có mục tiêu đã duyệt làm căn cứ)")

    source_map: dict[str, list] = {}
    all_parts: list[str] = list(obj_parts)
    global_idx = 0

    groups = [
        ("current_outcome", context_pack.current_outcome_contexts,
         "CĐR/PLO HIỆN HÀNH"),
        ("current_curriculum", context_pack.current_curriculum_contexts,
         "CẤU TRÚC CTĐT HIỆN HÀNH"),
        ("direction", context_pack.direction_contexts,
         "CHỈ ĐẠO / QUYẾT ĐỊNH CẬP NHẬT"),
        ("legal", context_pack.legal_contexts,
         "QUY ĐỊNH PHÁP LÝ"),
        ("evidence", context_pack.evidence_contexts,
         "KHẢO SÁT / BIÊN BẢN HỌP"),
        ("comparison", context_pack.comparison_contexts,
         "ĐỐI SÁNH CĐR"),
        ("course_syllabus", context_pack.course_syllabus_contexts,
         "ĐỀ CƯƠNG HỌC PHẦN"),
    ]

    for group_key, items, label in groups:
        if not items:
            all_parts.append(f"\n=== {label} ===\n(Không có dữ liệu)")
            continue
        all_parts.append(f"\n=== {label} ===")
        group_sources = []
        for item in items:
            text = item.text[:800] if item.text else "(trống)"
            fname = item.filename or "N/A"
            role = item.document_role or "unknown"
            all_parts.append(
                f"[Context {global_idx}] group={group_key} file={fname} "
                f"role={role} score={item.score:.3f}\n{text}"
            )
            group_sources.append({"global_index": global_idx, "item": item})
            global_idx += 1
        source_map[group_key] = group_sources

    if context_pack.missing_information:
        mi_lines = [f"- {m['type']}: {m['description']}" for m in context_pack.missing_information]
        all_parts.append("\n=== THÔNG TIN THIẾU ===\n" + "\n".join(mi_lines))

    contexts_block = "\n\n".join(all_parts) if all_parts else "(Không có contexts)"
    return f"{header}\n\n{contexts_block}", source_map


# ── Skill ────────────────────────────────────────────────────────────


class OutcomeUpdateSkill:
    """AI skill for proposing outcome (CĐR/PLO) updates."""

    async def run(self, *, update_cycle_id: str, program_id: str | None = None,
                  program_code: str | None = None, program_name: str | None = None,
                  context_pack, user_instruction: str | None = None,
                  outcome_count: int = 10,
                  group_allocation: dict[str, int] | None = None) -> OutcomeUpdateResult:
        total = (
            len(context_pack.current_outcome_contexts)
            + len(context_pack.current_curriculum_contexts)
            + len(context_pack.direction_contexts)
            + len(context_pack.legal_contexts)
            + len(context_pack.evidence_contexts)
            + len(context_pack.comparison_contexts)
            + len(context_pack.course_syllabus_contexts)
        )
        has_obj = context_pack.objective_update_payload is not None

        if total == 0 and not has_obj:
            return OutcomeUpdateResult(
                status=OutcomeUpdateStatus.INSUFFICIENT_CONTEXT,
                payload=OutcomeUpdatePayload(
                    outcome_update_strategy={"summary": "", "main_drivers": [], "human_review_required": True},
                    outcome_quality_review={
                        "overall_assessment": "Không đủ dữ liệu để đề xuất cập nhật chuẩn đầu ra.",
                        "strengths": [], "weaknesses": ["Thiếu tài liệu đầu vào."],
                        "consistency_notes": [],
                        "recommendation_for_human_review": ["Cần bổ sung tài liệu CTĐT và mục tiêu đào tạo."],
                    },
                    missing_information=list(context_pack.missing_information),
                ),
                warnings=["Không có contexts và không có objective_update_payload."],
            )

        if not getattr(settings, "SYNTHESIS_ENABLED", False):
            return OutcomeUpdateResult(
                status=OutcomeUpdateStatus.NEEDS_GENERATION,
                payload=OutcomeUpdatePayload(missing_information=list(context_pack.missing_information)),
                warnings=["SYNTHESIS_ENABLED=false. Cần bật LLM để sinh đề xuất CĐR.", f"Có {total} contexts sẵn sàng."],
            )

        api_key = getattr(settings, "OPENAI_API_KEY", None)
        if not api_key:
            return OutcomeUpdateResult(
                status=OutcomeUpdateStatus.NEEDS_GENERATION,
                payload=OutcomeUpdatePayload(missing_information=list(context_pack.missing_information)),
                warnings=["OPENAI_API_KEY chưa được cấu hình.", f"Có {total} contexts sẵn sàng."],
            )

        user_prompt, source_map = _build_user_prompt(
            program_name=program_name, program_code=program_code,
            update_cycle_id=update_cycle_id, context_pack=context_pack,
            user_instruction=user_instruction,
            outcome_count=outcome_count, group_allocation=group_allocation,
        )

        try:
            raw_json = await self._call_openai(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt, api_key=api_key)
        except Exception as exc:
            logger.warning("outcome_update_skill.llm_failed update_cycle=%s error=%s", update_cycle_id, exc.__class__.__name__)
            return OutcomeUpdateResult(
                status=OutcomeUpdateStatus.FAILED,
                payload=OutcomeUpdatePayload(missing_information=list(context_pack.missing_information)),
                warnings=[f"LLM gọi thất bại: {exc.__class__.__name__}", f"Có {total} contexts nhưng chưa thể phân tích."],
            )

        try:
            result = self._parse_response(raw_json, source_map, context_pack)
        except Exception as exc:
            logger.warning("outcome_update_skill.parse_failed update_cycle=%s error=%s", update_cycle_id, str(exc)[:200])
            return OutcomeUpdateResult(
                status=OutcomeUpdateStatus.FAILED,
                payload=OutcomeUpdatePayload(missing_information=list(context_pack.missing_information)),
                warnings=[f"Lỗi parse LLM response: {str(exc)[:150]}"],
            )

        # R6.8A-PATCH-1: post-process to enforce C-codes and allocation
        if result.status == OutcomeUpdateStatus.GENERATED:
            allocation = group_allocation or _compute_default_allocation(outcome_count)
            if result.payload.proposed_outcomes:
                result.payload.proposed_outcomes = _postprocess_outcomes(
                    result.payload.proposed_outcomes, outcome_count, allocation, result.warnings,
                )
            actual = len(result.payload.proposed_outcomes)
            # Quality gate: deficit → FAILED, truncated → keep GENERATED with warning
            if actual < outcome_count:
                result.status = OutcomeUpdateStatus.FAILED
                result.warnings.append(
                    f"AI chỉ sinh được {actual}/{outcome_count} chuẩn đầu ra hợp lệ. "
                    "Vui lòng sinh lại hoặc bổ sung thủ công."
                )

        logger.info("outcome_update_skill.done update_cycle=%s proposed=%d status=%s",
                     update_cycle_id, len(result.payload.proposed_outcomes), result.status)
        return result

    async def _call_openai(self, *, system_prompt: str, user_prompt: str, api_key: str) -> str:
        model = getattr(settings, "SYNTHESIS_MODEL", "gpt-4o-mini")
        timeout_s = float(getattr(settings, "SYNTHESIS_TIMEOUT_S", 90.0))
        max_tokens = int(getattr(settings, "SYNTHESIS_MAX_TOKENS", 8192))
        temperature = float(getattr(settings, "SYNTHESIS_TEMPERATURE", 0.15))
        payload = {
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "response_format": {"type": "json_object"},
        }
        async with httpx.AsyncClient(timeout=timeout_s + 2.0) as client:
            resp = await client.post("https://api.openai.com/v1/chat/completions",
                                     headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                                     json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty LLM response")
        return text.strip()

    def _parse_response(self, raw_json: str, source_map: dict[str, list], context_pack) -> OutcomeUpdateResult:
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")
        warnings: list[str] = []

        def _resolve_refs(raw_refs):
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

        def _s(val, default=""):
            return str(val).strip() if val else default

        strategy = data.get("outcome_update_strategy", {})
        if not isinstance(strategy, dict):
            strategy = {}

        current_analysis = []
        for raw in data.get("current_outcome_analysis", []):
            if not isinstance(raw, dict):
                continue
            current_analysis.append({
                "current_outcome": _s(raw.get("current_outcome")),
                "issue": _s(raw.get("issue")),
                "mapped_objectives": raw.get("mapped_objectives", []),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
                "confidence": _s(raw.get("confidence"), "medium"),
            })

        proposed = []
        for raw in data.get("proposed_outcomes", []):
            if not isinstance(raw, dict):
                continue
            refs = _resolve_refs(raw.get("evidence_refs"))
            flags = list(raw.get("quality_flags", []))
            confidence = _s(raw.get("confidence"), "medium")
            if not refs:
                confidence = "low"
                if "missing_evidence" not in flags:
                    flags.append("missing_evidence")
            code = raw.get("code")
            update_op = _norm_enum(raw.get("update_operation"), _ALLOWED_UPDATE_OP, "add", "update_operation", warnings)
            mapped_from_current = _s(raw.get("mapped_from_current"))
            # ── is_draft_code logic ──
            if code is None:
                is_draft = False
            elif update_op in ("add", "replace"):
                is_draft = True
            elif update_op in ("keep", "revise") and mapped_from_current:
                is_draft = raw.get("is_draft_code", False)
            else:
                is_draft = True
                if "needs_human_review" not in flags:
                    flags.append("needs_human_review")
            alignment = raw.get("alignment", {})
            if not isinstance(alignment, dict):
                alignment = {}
            mapped_objectives = raw.get("mapped_objectives", [])
            if not isinstance(mapped_objectives, list):
                mapped_objectives = []
            if not mapped_objectives:
                if "missing_objective_mapping" not in flags:
                    flags.append("missing_objective_mapping")
                if confidence != "low":
                    confidence = "low"
            proposed_content = _s(raw.get("proposed_content"))
            outcome_type = _norm_enum(raw.get("outcome_type"), _ALLOWED_OUTCOME_TYPE, "other", "outcome_type", warnings)
            if _looks_too_broad(proposed_content):
                for f in ("overlaps_with_objective", "too_broad", "needs_human_review"):
                    if f not in flags:
                        flags.append(f)
            if _looks_too_course_specific(proposed_content):
                for f in ("too_course_specific", "needs_human_review"):
                    if f not in flags:
                        flags.append(f)
            bloom = _norm_enum(raw.get("bloom_level"), _ALLOWED_BLOOM, "unknown", "bloom_level", warnings)
            priority = _norm_enum(raw.get("priority"), _ALLOWED_PRIORITY, "medium", "priority", warnings)
            confidence = _norm_enum(confidence, _ALLOWED_CONFIDENCE, "medium", "confidence", warnings)
            flags = _norm_quality_flags(flags, warnings)
            proposed.append({
                "outcome_type": outcome_type, "code": code, "is_draft_code": is_draft,
                "update_operation": update_op,
                "mapped_from_current": mapped_from_current,
                "proposed_content": proposed_content, "rationale": _s(raw.get("rationale")),
                "bloom_level": bloom,
                "mapped_objectives": mapped_objectives, "alignment": alignment,
                "evidence_refs": refs, "quality_flags": flags,
                "priority": priority, "confidence": confidence,
            })

        obj_outcome_alignment = []
        for raw in data.get("objective_outcome_alignment", []):
            if not isinstance(raw, dict):
                continue
            obj_outcome_alignment.append({
                "objective_code": _s(raw.get("objective_code")),
                "objective_content": _s(raw.get("objective_content")),
                "mapped_outcomes": raw.get("mapped_outcomes", []),
                "coverage_status": _norm_enum(raw.get("coverage_status"), _ALLOWED_COVERAGE, "unknown", "coverage_status", warnings),
                "notes": _s(raw.get("notes")),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
            })

        quality_review = data.get("outcome_quality_review", {})
        if not isinstance(quality_review, dict):
            quality_review = {}
        missing_info = []
        for raw in data.get("missing_information", []):
            if not isinstance(raw, dict):
                continue
            missing_info.append({"type": _s(raw.get("type"), "other"), "description": _s(raw.get("description"))})
        risks = []
        for raw in data.get("risks", []):
            if not isinstance(raw, dict):
                continue
            risks.append({"risk": _s(raw.get("risk")), "impact": _s(raw.get("impact"), "medium"), "mitigation": _s(raw.get("mitigation"))})
        next_actions = []
        for raw in data.get("next_actions", []):
            if not isinstance(raw, dict):
                continue
            next_actions.append({"action": _s(raw.get("action")), "owner_hint": _s(raw.get("owner_hint"), "unknown"), "priority": _s(raw.get("priority"), "medium")})

        payload = OutcomeUpdatePayload(
            outcome_update_strategy=strategy, current_outcome_analysis=current_analysis,
            proposed_outcomes=proposed, objective_outcome_alignment=obj_outcome_alignment,
            outcome_quality_review=quality_review, missing_information=missing_info,
            risks=risks, next_actions=next_actions,
        )
        status = OutcomeUpdateStatus.GENERATED if proposed or current_analysis else OutcomeUpdateStatus.INSUFFICIENT_CONTEXT
        return OutcomeUpdateResult(status=status, payload=payload, warnings=warnings)
