"""
CurriculumUpdateDesignSkill — Đề xuất cập nhật chương trình đào tạo mới.

R6.0: Đọc yêu cầu nhà trường + CTĐT cũ + minh chứng → đề xuất cải tiến
CTĐT mới gồm mục tiêu, CĐR, cấu trúc, học phần, ma trận.

Đây KHÔNG phải Mẫu 06. Đây là bản nháp thiết kế CTĐT mới.

Architecture::

    run()
        ├─ no contexts? → insufficient_evidence + missing_information
        ├─ LLM disabled? → needs_generation (deterministic)
        ├─ build prompt (curriculum design specific)
        ├─ call OpenAI
        ├─ parse JSON response → CurriculumUpdateDesignResult
        └─ validate: each item's evidence_refs must exist in input

Guards:
    - Không ghi vào Program / ProgramVersion / ProgramVersionRevision.
    - Không tự động cập nhật dữ liệu CTĐT chính thức.
    - Output là JSON có cấu trúc, chỉ là bản nháp đề xuất.
    - Nếu thiếu tài liệu thì đưa vào missing_information, không bịa.
    - evidence_refs phải có nguồn thực từ input sources.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from app.core.config import settings
from app.services.ctdt_analysis_service import AnalysisSource

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────


class DesignStatus(str, Enum):
    GENERATED = "generated"
    NEEDS_GENERATION = "needs_generation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CourseAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    REMOVE = "remove"
    MERGE = "merge"
    SPLIT = "split"
    KEEP = "keep"


class CurriculumArea(str, Enum):
    GENERAL = "general"
    FOUNDATION = "foundation"
    MAJOR = "major"
    INTERNSHIP = "internship"
    GRADUATION = "graduation"
    OTHER = "other"


class MatrixLevel(str, Enum):
    I = "I"
    R = "R"
    M = "M"
    UNKNOWN = "unknown"


class MissingInfoType(str, Enum):
    CURRENT_CURRICULUM = "current_curriculum"
    DIRECTION_DECISION = "direction_decision"
    LEGAL_REGULATION = "legal_regulation"
    SURVEY_EVIDENCE = "survey_evidence"
    COURSE_SYLLABUS = "course_syllabus"
    MATRIX = "matrix"
    OTHER = "other"


class RiskImpact(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class OwnerHint(str, Enum):
    BO_MON = "bo_mon"
    KHOA = "khoa"
    HOI_DONG = "hoi_dong"
    ADMIN = "admin"
    UNKNOWN = "unknown"


# ── Result DTOs ──────────────────────────────────────────────────────


@dataclass
class EvidenceRef:
    """A resolved source reference."""
    source_index: int
    ai_document_id: int
    filename: str | None
    document_role: str | None
    chunk_index: int
    score: float
    quote: str


@dataclass
class CurriculumUpdateDesignPayload:
    """Full payload matching the R6.0 output JSON schema."""
    update_orientation: str = ""
    objective_updates: list[dict[str, Any]] = field(default_factory=list)
    outcome_updates: list[dict[str, Any]] = field(default_factory=list)
    curriculum_structure_updates: list[dict[str, Any]] = field(default_factory=list)
    course_updates: list[dict[str, Any]] = field(default_factory=list)
    matrix_update_notes: list[dict[str, Any]] = field(default_factory=list)
    evidence_based_rationale: list[dict[str, Any]] = field(default_factory=list)
    missing_information: list[dict[str, Any]] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    next_actions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CurriculumUpdateDesignResult:
    """Full result from curriculum update design skill."""
    status: str  # DesignStatus value
    payload: CurriculumUpdateDesignPayload
    warnings: list[str]
    task_type: str = "curriculum_update_design"


# ── Prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Bạn là chuyên gia tư vấn thiết kế và cập nhật chương trình đào tạo đại học Việt Nam.

NHIỆM VỤ: Dựa trên CTĐT hiện hành, các minh chứng đánh giá, yêu cầu/chỉ đạo \
của nhà trường và quy định pháp lý → ĐỀ XUẤT bản nháp cập nhật CTĐT mới, gồm:
- Định hướng cập nhật tổng thể
- Cập nhật mục tiêu đào tạo
- Cập nhật chuẩn đầu ra (CĐR) với bậc Bloom
- Cập nhật cấu trúc chương trình
- Cập nhật học phần (thêm/sửa/xoá/gộp/tách/giữ)
- Ghi chú ma trận CĐR-học phần
- Phân tích rủi ro và hành động tiếp theo

QUY TẮC BẮT BUỘC:
1. CHỈ đề xuất dựa trên các đoạn trích (contexts) được cung cấp.
2. KHÔNG bịa thêm thông tin, số liệu, mã học phần, số tín chỉ, mã CĐR, \
hoặc nguồn không có trong contexts.
3. KHÔNG tự ghi vào CTĐT chính thức. Đây chỉ là BẢN NHÁP đề xuất.
4. Mỗi đề xuất phải gắn source_indices (chỉ số 0-indexed) trong evidence_refs.
5. Nếu không đủ căn cứ, đặt confidence = "low" và ghi vào missing_information.
6. Nếu thiếu tài liệu CTĐT hiện hành, ghi missing_information type="current_curriculum".
7. Nếu thiếu quyết định chỉ đạo, ghi missing_information type="direction_decision".

OUTPUT FORMAT (JSON):
{
  "update_orientation": "Mô tả ngắn gọn định hướng cập nhật tổng thể (2-3 câu)",
  "objective_updates": [
    {
      "current_objective": "Mục tiêu hiện tại (nếu biết)",
      "proposed_objective": "Mục tiêu đề xuất mới",
      "reason": "Lý do thay đổi",
      "evidence_refs": [{"source_index": 0}],
      "priority": "low|medium|high",
      "confidence": "low|medium|high"
    }
  ],
  "outcome_updates": [
    {
      "current_outcome": "CĐR hiện tại",
      "proposed_outcome": "CĐR đề xuất mới",
      "bloom_level": "Bậc Bloom (remember/understand/apply/analyze/evaluate/create)",
      "reason": "Lý do",
      "evidence_refs": [{"source_index": 0}],
      "priority": "low|medium|high",
      "confidence": "low|medium|high"
    }
  ],
  "curriculum_structure_updates": [
    {
      "area": "general|foundation|major|internship|graduation|other",
      "current_state": "Tình trạng hiện tại",
      "proposed_change": "Đề xuất thay đổi",
      "reason": "Lý do",
      "evidence_refs": [{"source_index": 0}],
      "priority": "low|medium|high",
      "confidence": "low|medium|high"
    }
  ],
  "course_updates": [
    {
      "course_code": null,
      "course_name": "Tên học phần",
      "action": "add|update|remove|merge|split|keep",
      "current_state": "Tình trạng hiện tại",
      "proposed_change": "Đề xuất thay đổi",
      "reason": "Lý do",
      "related_outcomes": [],
      "evidence_refs": [{"source_index": 0}],
      "priority": "low|medium|high",
      "confidence": "low|medium|high"
    }
  ],
  "matrix_update_notes": [
    {
      "outcome": "CĐR liên quan",
      "course": "Học phần",
      "suggested_level": "I|R|M|unknown",
      "reason": "Lý do",
      "evidence_refs": [{"source_index": 0}]
    }
  ],
  "evidence_based_rationale": [
    {
      "claim": "Nhận định dựa trên minh chứng",
      "evidence_refs": [{"source_index": 0}]
    }
  ],
  "missing_information": [
    {
      "type": "current_curriculum|direction_decision|legal_regulation|survey_evidence|course_syllabus|matrix|other",
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
    sources: list[AnalysisSource],
    user_instruction: str | None = None,
) -> str:
    """Build user prompt with contexts from retrieval."""
    header = f"Đợt cập nhật CTĐT: {update_cycle_id}"
    if program_name:
        header += f"\nChương trình: {program_name}"
    if program_code:
        header += f" (Mã ngành: {program_code})"

    if user_instruction:
        header += f"\n\nHướng dẫn bổ sung từ hệ thống: {user_instruction}"

    context_parts = []
    for i, src in enumerate(sources):
        role_label = src.document_role or "unknown"
        fname = src.filename or "N/A"
        text = src.quote[:800] if src.quote else "(trống)"
        context_parts.append(
            f"[Context {i}] file={fname} role={role_label} "
            f"score={src.score:.3f}\n{text}"
        )

    contexts_block = "\n\n".join(context_parts) if context_parts else "(Không có contexts)"

    return f"{header}\n\nCác đoạn trích từ tài liệu:\n\n{contexts_block}"


# ── Skill ────────────────────────────────────────────────────────────


class CurriculumUpdateDesignSkill:
    """
    AI skill for drafting curriculum update design proposals.

    Fail-open design: LLM errors produce status=failed with warnings,
    never crash the calling pipeline.

    Guards:
      - No writes to Program/ProgramVersion/ProgramVersionRevision.
      - Output is a structured JSON draft only.
      - Missing documents → missing_information, never fabrication.
    """

    async def run(
        self,
        *,
        update_cycle_id: str,
        program_id: str | None = None,
        program_code: str | None = None,
        program_name: str | None = None,
        sources: list[AnalysisSource],
        user_instruction: str | None = None,
    ) -> CurriculumUpdateDesignResult:
        """
        Run curriculum update design drafting on the provided sources.

        Returns:
          - insufficient_evidence if no sources
          - needs_generation if LLM is disabled
          - generated if LLM succeeds
          - failed if LLM errors
        """
        # ── No contexts → insufficient_evidence ──────────────────
        if not sources:
            return CurriculumUpdateDesignResult(
                status=DesignStatus.INSUFFICIENT_EVIDENCE.value,
                payload=CurriculumUpdateDesignPayload(
                    update_orientation="",
                    missing_information=[
                        {
                            "type": "current_curriculum",
                            "description": "Không có tài liệu CTĐT hiện hành trong phạm vi đợt cập nhật.",
                        },
                        {
                            "type": "direction_decision",
                            "description": "Không có quyết định/chỉ đạo cập nhật CTĐT.",
                        },
                    ],
                ),
                warnings=["Không có tài liệu nào được tìm thấy trong phạm vi đợt cập nhật."],
            )

        # ── LLM disabled → deterministic fallback ────────────────
        if not getattr(settings, "SYNTHESIS_ENABLED", False):
            return CurriculumUpdateDesignResult(
                status=DesignStatus.NEEDS_GENERATION.value,
                payload=CurriculumUpdateDesignPayload(),
                warnings=[
                    "SYNTHESIS_ENABLED=false. Cần bật LLM để sinh đề xuất cập nhật CTĐT.",
                    f"Có {len(sources)} nguồn tài liệu sẵn sàng để phân tích.",
                ],
            )

        # ── Check OpenAI key ─────────────────────────────────────
        api_key = getattr(settings, "OPENAI_API_KEY", None)
        if not api_key:
            return CurriculumUpdateDesignResult(
                status=DesignStatus.NEEDS_GENERATION.value,
                payload=CurriculumUpdateDesignPayload(),
                warnings=[
                    "OPENAI_API_KEY chưa được cấu hình.",
                    f"Có {len(sources)} nguồn tài liệu sẵn sàng để phân tích.",
                ],
            )

        # ── Build prompt ─────────────────────────────────────────
        user_prompt = _build_user_prompt(
            program_name=program_name,
            program_code=program_code,
            update_cycle_id=update_cycle_id,
            sources=sources,
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
                "curriculum_update_design_skill.llm_failed update_cycle=%s error=%s",
                update_cycle_id, exc.__class__.__name__,
            )
            return CurriculumUpdateDesignResult(
                status=DesignStatus.FAILED.value,
                payload=CurriculumUpdateDesignPayload(),
                warnings=[
                    f"LLM gọi thất bại: {exc.__class__.__name__}",
                    f"Có {len(sources)} nguồn tài liệu nhưng chưa thể phân tích.",
                ],
            )

        # ── Parse response ───────────────────────────────────────
        try:
            result = self._parse_response(raw_json, sources)
        except Exception as exc:
            logger.warning(
                "curriculum_update_design_skill.parse_failed update_cycle=%s error=%s",
                update_cycle_id, str(exc)[:200],
            )
            return CurriculumUpdateDesignResult(
                status=DesignStatus.FAILED.value,
                payload=CurriculumUpdateDesignPayload(),
                warnings=[
                    f"Lỗi parse LLM response: {str(exc)[:150]}",
                ],
            )

        logger.info(
            "curriculum_update_design_skill.done update_cycle=%s "
            "objectives=%d outcomes=%d courses=%d status=%s",
            update_cycle_id,
            len(result.payload.objective_updates),
            len(result.payload.outcome_updates),
            len(result.payload.course_updates),
            result.status,
        )
        return result

    # ── OpenAI call ──────────────────────────────────────────────

    async def _call_openai(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        api_key: str,
    ) -> str:
        """Call OpenAI and return raw text response."""
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
        self,
        raw_json: str,
        input_sources: list[AnalysisSource],
    ) -> CurriculumUpdateDesignResult:
        """Parse LLM JSON response and validate source references."""
        data = json.loads(raw_json)

        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")

        warnings: list[str] = []

        # ── update_orientation ───────────────────────────────────
        update_orientation = str(data.get("update_orientation", "")).strip()

        # ── Helper: resolve evidence_refs ────────────────────────
        def _resolve_refs(raw_refs: list | None) -> list[dict[str, Any]]:
            if not raw_refs or not isinstance(raw_refs, list):
                return []
            resolved = []
            for ref in raw_refs:
                if not isinstance(ref, dict):
                    continue
                idx = ref.get("source_index")
                if isinstance(idx, int) and 0 <= idx < len(input_sources):
                    src = input_sources[idx]
                    resolved.append({
                        "source_index": idx,
                        "ai_document_id": src.ai_document_id,
                        "filename": src.filename,
                        "document_role": src.document_role,
                        "chunk_index": src.chunk_index,
                        "score": src.score,
                        "quote": (src.quote[:200] if src.quote else ""),
                    })
                elif isinstance(idx, int):
                    warnings.append(
                        f"source_index {idx} out of range (0–{len(input_sources)-1})"
                    )
            return resolved

        # ── Helper: validate enum ────────────────────────────────
        def _safe_enum(val: Any, enum_cls: type[Enum], default: str) -> str:
            try:
                enum_cls(val)
                return str(val)
            except (ValueError, KeyError):
                return default

        # ── Parse arrays ─────────────────────────────────────────
        objective_updates = []
        for raw in data.get("objective_updates", []):
            if not isinstance(raw, dict):
                continue
            objective_updates.append({
                "current_objective": str(raw.get("current_objective", "")).strip(),
                "proposed_objective": str(raw.get("proposed_objective", "")).strip(),
                "reason": str(raw.get("reason", "")).strip(),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
                "priority": _safe_enum(raw.get("priority"), Priority, "medium"),
                "confidence": _safe_enum(raw.get("confidence"), Confidence, "medium"),
            })

        outcome_updates = []
        for raw in data.get("outcome_updates", []):
            if not isinstance(raw, dict):
                continue
            outcome_updates.append({
                "current_outcome": str(raw.get("current_outcome", "")).strip(),
                "proposed_outcome": str(raw.get("proposed_outcome", "")).strip(),
                "bloom_level": str(raw.get("bloom_level", "")).strip(),
                "reason": str(raw.get("reason", "")).strip(),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
                "priority": _safe_enum(raw.get("priority"), Priority, "medium"),
                "confidence": _safe_enum(raw.get("confidence"), Confidence, "medium"),
            })

        curriculum_structure_updates = []
        for raw in data.get("curriculum_structure_updates", []):
            if not isinstance(raw, dict):
                continue
            curriculum_structure_updates.append({
                "area": _safe_enum(raw.get("area"), CurriculumArea, "other"),
                "current_state": str(raw.get("current_state", "")).strip(),
                "proposed_change": str(raw.get("proposed_change", "")).strip(),
                "reason": str(raw.get("reason", "")).strip(),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
                "priority": _safe_enum(raw.get("priority"), Priority, "medium"),
                "confidence": _safe_enum(raw.get("confidence"), Confidence, "medium"),
            })

        course_updates = []
        for raw in data.get("course_updates", []):
            if not isinstance(raw, dict):
                continue
            course_updates.append({
                "course_code": raw.get("course_code"),
                "course_name": str(raw.get("course_name", "")).strip(),
                "action": _safe_enum(raw.get("action"), CourseAction, "keep"),
                "current_state": str(raw.get("current_state", "")).strip(),
                "proposed_change": str(raw.get("proposed_change", "")).strip(),
                "reason": str(raw.get("reason", "")).strip(),
                "related_outcomes": list(raw.get("related_outcomes", [])),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
                "priority": _safe_enum(raw.get("priority"), Priority, "medium"),
                "confidence": _safe_enum(raw.get("confidence"), Confidence, "medium"),
            })

        matrix_update_notes = []
        for raw in data.get("matrix_update_notes", []):
            if not isinstance(raw, dict):
                continue
            matrix_update_notes.append({
                "outcome": str(raw.get("outcome", "")).strip(),
                "course": str(raw.get("course", "")).strip(),
                "suggested_level": _safe_enum(
                    raw.get("suggested_level"), MatrixLevel, "unknown",
                ),
                "reason": str(raw.get("reason", "")).strip(),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
            })

        evidence_based_rationale = []
        for raw in data.get("evidence_based_rationale", []):
            if not isinstance(raw, dict):
                continue
            evidence_based_rationale.append({
                "claim": str(raw.get("claim", "")).strip(),
                "evidence_refs": _resolve_refs(raw.get("evidence_refs")),
            })

        missing_information = []
        for raw in data.get("missing_information", []):
            if not isinstance(raw, dict):
                continue
            missing_information.append({
                "type": _safe_enum(raw.get("type"), MissingInfoType, "other"),
                "description": str(raw.get("description", "")).strip(),
            })

        risks = []
        for raw in data.get("risks", []):
            if not isinstance(raw, dict):
                continue
            risks.append({
                "risk": str(raw.get("risk", "")).strip(),
                "impact": _safe_enum(raw.get("impact"), RiskImpact, "medium"),
                "mitigation": str(raw.get("mitigation", "")).strip(),
            })

        next_actions = []
        for raw in data.get("next_actions", []):
            if not isinstance(raw, dict):
                continue
            next_actions.append({
                "action": str(raw.get("action", "")).strip(),
                "owner_hint": _safe_enum(raw.get("owner_hint"), OwnerHint, "unknown"),
                "priority": _safe_enum(raw.get("priority"), Priority, "medium"),
            })

        payload = CurriculumUpdateDesignPayload(
            update_orientation=update_orientation,
            objective_updates=objective_updates,
            outcome_updates=outcome_updates,
            curriculum_structure_updates=curriculum_structure_updates,
            course_updates=course_updates,
            matrix_update_notes=matrix_update_notes,
            evidence_based_rationale=evidence_based_rationale,
            missing_information=missing_information,
            risks=risks,
            next_actions=next_actions,
        )

        status = (
            DesignStatus.GENERATED.value
            if (objective_updates or outcome_updates or course_updates
                or curriculum_structure_updates)
            else DesignStatus.INSUFFICIENT_EVIDENCE.value
        )

        return CurriculumUpdateDesignResult(
            status=status,
            payload=payload,
            warnings=warnings,
        )
