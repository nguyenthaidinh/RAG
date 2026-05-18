"""
EvidenceAnalysisSkill — Phân tích minh chứng phục vụ Mẫu 04.

Mẫu 04: "Báo cáo thông tin, minh chứng liên quan đến sự cần thiết
phải đánh giá, cải tiến chương trình đào tạo."

R5.1/R5.2:
- Nhận contexts/sources từ R4 evidence_analysis retrieval task.
- Nếu SYNTHESIS_ENABLED=true → gọi OpenAI để phân tích minh chứng.
- Nếu SYNTHESIS_ENABLED=false → trả deterministic fallback.
- Không bịa nội dung, không tự tạo nguồn giả.
- Không yêu cầu AI sửa CTĐT hay sinh mục tiêu/CĐR.

Architecture::

    run()
        ├─ no contexts? → insufficient_evidence
        ├─ LLM disabled? → needs_generation (deterministic)
        ├─ build prompt (Mẫu 04 specific)
        ├─ call OpenAI
        ├─ parse JSON response → EvidenceAnalysisResult
        └─ validate: each item's sources must exist in input
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import httpx

from app.core.config import settings
from app.services.ctdt_analysis_service import AnalysisSource

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────


class EvidenceType(str, Enum):
    SURVEY = "survey"
    REGULATION = "regulation"
    MEETING = "meeting"
    DECISION = "decision"
    COMPARISON = "comparison"
    CURRENT_CURRICULUM = "current_curriculum"
    OTHER = "other"


class EvidenceConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class EvidenceStatus(str, Enum):
    GENERATED = "generated"
    NEEDS_GENERATION = "needs_generation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


# ── Result DTOs ──────────────────────────────────────────────────────


@dataclass
class EvidenceSummaryItem:
    """One evidence item in the analysis."""
    title: str
    summary: str
    evidence_type: str  # EvidenceType value
    rationale: str
    confidence: str  # EvidenceConfidence value
    sources: list[AnalysisSource]


@dataclass
class EvidenceAnalysisResult:
    """Full result from evidence analysis skill."""
    status: str  # EvidenceStatus value
    items: list[EvidenceSummaryItem]
    warnings: list[str]
    task_type: str = "evidence_analysis"


# ── Prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Bạn là chuyên gia phân tích chương trình đào tạo đại học Việt Nam.

NHIỆM VỤ: Phân tích các minh chứng, thông tin liên quan đến sự cần thiết \
phải đánh giá, cải tiến chương trình đào tạo (phục vụ Mẫu 04).

QUY TẮC BẮT BUỘC:
1. CHỈ phân tích và tổng hợp từ các đoạn trích (contexts) được cung cấp.
2. KHÔNG bịa thêm thông tin, số liệu, hoặc nguồn không có trong contexts.
3. KHÔNG đề xuất sửa đổi CTĐT, KHÔNG sinh mục tiêu/chuẩn đầu ra.
4. Mỗi item phải gắn source_indices (chỉ số 0-indexed) tham chiếu đến contexts đầu vào.
5. Nếu không đủ căn cứ cho một loại minh chứng, đặt confidence = "insufficient_evidence".

CÁC LOẠI MINH CHỨNG CẦN PHÂN TÍCH:
- survey: Khảo sát sinh viên, cựu sinh viên, nhà tuyển dụng, giảng viên
- regulation: Văn bản pháp lý, thông tư, quy định (TT17, TT07, Luật GD...)
- meeting: Biên bản họp, phản biện, hội thảo, góp ý
- decision: Quyết định, chỉ đạo của cấp trên
- comparison: Báo cáo đối sánh với CTĐT trong/ngoài nước
- current_curriculum: Thông tin CTĐT hiện hành cần đánh giá
- other: Minh chứng khác

OUTPUT FORMAT (JSON):
{
  "items": [
    {
      "title": "Tiêu đề ngắn gọn",
      "summary": "Tóm tắt nội dung minh chứng (2-4 câu)",
      "evidence_type": "survey|regulation|meeting|decision|comparison|current_curriculum|other",
      "rationale": "Giải thích tại sao minh chứng này cho thấy sự cần thiết cập nhật CTĐT",
      "confidence": "high|medium|low|insufficient_evidence",
      "source_indices": [0, 1]
    }
  ],
  "warnings": ["Cảnh báo nếu có"]
}

Chỉ trả JSON, không giải thích thêm.\
"""


def _build_user_prompt(
    *,
    program_name: str | None,
    program_code: str | None,
    update_cycle_id: str,
    sources: list[AnalysisSource],
) -> str:
    """Build user prompt with contexts from retrieval."""
    header = f"Đợt cập nhật CTĐT: {update_cycle_id}"
    if program_name:
        header += f"\nChương trình: {program_name}"
    if program_code:
        header += f" (Mã ngành: {program_code})"

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


class EvidenceAnalysisSkill:
    """
    AI skill for analyzing evidence supporting CTĐT update necessity.

    Fail-open design: LLM errors produce status=failed with warnings,
    never crash the calling analysis pipeline.
    """

    async def run(
        self,
        *,
        update_cycle_id: str,
        program_code: str | None = None,
        program_name: str | None = None,
        sources: list[AnalysisSource],
    ) -> EvidenceAnalysisResult:
        """
        Run evidence analysis on the provided sources.

        Returns:
          - insufficient_evidence if no sources
          - needs_generation if LLM is disabled
          - generated if LLM succeeds
          - failed if LLM errors
        """
        # ── No contexts → insufficient_evidence ──────────────────
        if not sources:
            return EvidenceAnalysisResult(
                status=EvidenceStatus.INSUFFICIENT_EVIDENCE.value,
                items=[],
                warnings=["Không có minh chứng nào được tìm thấy trong phạm vi đợt cập nhật."],
            )

        # ── LLM disabled → deterministic fallback ────────────────
        if not getattr(settings, "SYNTHESIS_ENABLED", False):
            return EvidenceAnalysisResult(
                status=EvidenceStatus.NEEDS_GENERATION.value,
                items=[],
                warnings=[
                    "SYNTHESIS_ENABLED=false. Cần bật LLM để sinh phân tích minh chứng.",
                    f"Có {len(sources)} nguồn minh chứng sẵn sàng để phân tích.",
                ],
            )

        # ── Check OpenAI key ─────────────────────────────────────
        api_key = getattr(settings, "OPENAI_API_KEY", None)
        if not api_key:
            return EvidenceAnalysisResult(
                status=EvidenceStatus.NEEDS_GENERATION.value,
                items=[],
                warnings=[
                    "OPENAI_API_KEY chưa được cấu hình.",
                    f"Có {len(sources)} nguồn minh chứng sẵn sàng để phân tích.",
                ],
            )

        # ── Build prompt ─────────────────────────────────────────
        user_prompt = _build_user_prompt(
            program_name=program_name,
            program_code=program_code,
            update_cycle_id=update_cycle_id,
            sources=sources,
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
                "evidence_skill.llm_failed update_cycle=%s error=%s",
                update_cycle_id, exc.__class__.__name__,
            )
            return EvidenceAnalysisResult(
                status=EvidenceStatus.FAILED.value,
                items=[],
                warnings=[
                    f"LLM gọi thất bại: {exc.__class__.__name__}",
                    f"Có {len(sources)} nguồn minh chứng nhưng chưa thể phân tích.",
                ],
            )

        # ── Parse response ───────────────────────────────────────
        try:
            result = self._parse_response(raw_json, sources)
        except Exception as exc:
            logger.warning(
                "evidence_skill.parse_failed update_cycle=%s error=%s",
                update_cycle_id, str(exc)[:200],
            )
            return EvidenceAnalysisResult(
                status=EvidenceStatus.FAILED.value,
                items=[],
                warnings=[
                    f"Lỗi parse LLM response: {str(exc)[:150]}",
                ],
            )

        logger.info(
            "evidence_skill.done update_cycle=%s items=%d status=%s",
            update_cycle_id, len(result.items), result.status,
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
        timeout_s = float(getattr(settings, "SYNTHESIS_TIMEOUT_S", 60.0))
        max_tokens = int(getattr(settings, "SYNTHESIS_MAX_TOKENS", 4096))
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
    ) -> EvidenceAnalysisResult:
        """Parse LLM JSON response and validate source references."""
        data = json.loads(raw_json)

        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")

        raw_items = data.get("items", [])
        warnings = list(data.get("warnings", []))

        if not isinstance(raw_items, list):
            raise ValueError("'items' is not a list")

        items: list[EvidenceSummaryItem] = []

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue

            # Validate evidence_type
            ev_type = raw.get("evidence_type", "other")
            try:
                EvidenceType(ev_type)
            except ValueError:
                ev_type = "other"

            # Validate confidence
            confidence = raw.get("confidence", "medium")
            try:
                EvidenceConfidence(confidence)
            except ValueError:
                confidence = "medium"

            # Resolve source_indices to actual AnalysisSource objects
            source_indices = raw.get("source_indices", [])
            resolved_sources: list[AnalysisSource] = []
            for idx in source_indices:
                if isinstance(idx, int) and 0 <= idx < len(input_sources):
                    resolved_sources.append(input_sources[idx])
                else:
                    warnings.append(
                        f"source_index {idx} out of range (0–{len(input_sources)-1})"
                    )

            items.append(EvidenceSummaryItem(
                title=str(raw.get("title", "")).strip() or "Minh chứng",
                summary=str(raw.get("summary", "")).strip(),
                evidence_type=ev_type,
                rationale=str(raw.get("rationale", "")).strip(),
                confidence=confidence,
                sources=resolved_sources,
            ))

        status = (
            EvidenceStatus.GENERATED.value
            if items
            else EvidenceStatus.INSUFFICIENT_EVIDENCE.value
        )

        return EvidenceAnalysisResult(
            status=status,
            items=items,
            warnings=warnings,
        )
