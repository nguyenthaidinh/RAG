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
    """Full payload matching the R6.1B output JSON schema."""
    objective_update_strategy: dict[str, Any] = field(default_factory=dict)
    current_objective_analysis: list[dict[str, Any]] = field(default_factory=list)
    proposed_objectives: list[dict[str, Any]] = field(default_factory=list)
    alignment_notes: list[dict[str, Any]] = field(default_factory=list)
    objective_quality_review: dict[str, Any] = field(default_factory=dict)
    missing_information: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    next_actions: list[dict[str, Any]] = field(default_factory=list)


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
khảo sát bên liên quan, và đối sánh → ĐỀ XUẤT cập nhật mục tiêu đào tạo.

MỤC TIÊU ĐÀO TẠO LÀ GÌ:
- Là định hướng CẤP CHƯƠNG TRÌNH (không phải cấp học phần).
- Mô tả người học sau tốt nghiệp có thể phát triển theo HƯỚNG NÀO.
- Thể hiện năng lực nghề nghiệp, phẩm chất, vai trò xã hội ở MỨC TỔNG QUÁT.
- Là nền để dẫn xuất chuẩn đầu ra (CĐR), KHÔNG PHẢI CĐR.

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


def _build_user_prompt(
    *,
    program_name: str | None,
    program_code: str | None,
    update_cycle_id: str,
    context_pack,
    user_instruction: str | None = None,
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
        user_prompt, source_map = _build_user_prompt(
            program_name=program_name,
            program_code=program_code,
            update_cycle_id=update_cycle_id,
            context_pack=context_pack,
            user_instruction=user_instruction,
        )

        # ── Call LLM ─────────────────────────────────────────────
        try:
            raw_json = await self._call_openai(
                system_prompt=_SYSTEM_PROMPT,
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
            result = self._parse_response(raw_json, source_map, context_pack)
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

        payload = ObjectiveUpdatePayload(
            objective_update_strategy=strategy,
            current_objective_analysis=current_analysis,
            proposed_objectives=proposed,
            alignment_notes=alignment_notes,
            objective_quality_review=quality_review,
            missing_information=missing_info,
            risks=risks,
            next_actions=next_actions,
        )

        status = (
            ObjectiveUpdateStatus.GENERATED
            if proposed or current_analysis
            else ObjectiveUpdateStatus.INSUFFICIENT_CONTEXT
        )

        return ObjectiveUpdateResult(
            status=status, payload=payload, warnings=warnings,
        )
