"""
CurrentCurriculumReviewSkill — Đánh giá CTĐT hiện hành phục vụ Mẫu 05.

Mẫu 05: "Báo cáo đánh giá tính hiệu quả của chương trình đào tạo
đang thực hiện."

R5.3:
- Nhận contexts/sources từ R4 current_curriculum_review retrieval task.
- Nếu SYNTHESIS_ENABLED=true → gọi OpenAI để đánh giá CTĐT hiện hành.
- Nếu SYNTHESIS_ENABLED=false → trả deterministic fallback.
- Không bịa nội dung, không tự tạo nguồn giả.
- Không sửa CTĐT, không sinh mục tiêu/CĐR mới.

Architecture::

    run()
        ├─ no contexts? → insufficient_evidence
        ├─ LLM disabled? → needs_generation (deterministic)
        ├─ build prompt (Mẫu 05 specific)
        ├─ call OpenAI
        ├─ parse JSON response → CurrentCurriculumReviewResult
        └─ validate: each item's sources must exist in input
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx

from app.core.config import settings
from app.services.ctdt_analysis_service import AnalysisSource

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────


class ReviewConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ReviewStatus(str, Enum):
    GENERATED = "generated"
    NEEDS_GENERATION = "needs_generation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


# ── Result DTOs ──────────────────────────────────────────────────────


@dataclass
class CurriculumEvaluationPoint:
    """One evaluation point about the current curriculum."""
    aspect: str
    finding: str
    recommendation: str
    rationale: str
    confidence: str  # ReviewConfidence value
    sources: list[AnalysisSource]


@dataclass
class CurrentCurriculumReviewResult:
    """Full result from curriculum review skill."""
    status: str  # ReviewStatus value
    items: list[CurriculumEvaluationPoint]
    warnings: list[str]
    task_type: str = "current_curriculum_review"


# ── Prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Bạn là chuyên gia đánh giá chương trình đào tạo đại học Việt Nam.

NHIỆM VỤ: Đánh giá tính hiệu quả của chương trình đào tạo (CTĐT) đang \
thực hiện, dựa trên các đoạn trích tài liệu được cung cấp (phục vụ Mẫu 05).

QUY TẮC BẮT BUỘC:
1. CHỈ đánh giá dựa trên các đoạn trích (contexts) được cung cấp.
2. KHÔNG bịa thêm thông tin, số liệu, tên học phần, mã CĐR, hoặc nguồn \
không có trong contexts.
3. KHÔNG sửa đổi CTĐT, KHÔNG sinh mục tiêu mới, KHÔNG sinh chuẩn đầu ra mới, \
KHÔNG thay đổi học phần, KHÔNG xuất file.
4. Chỉ đánh giá và đề xuất hướng cải tiến ở mức nháp dựa trên minh chứng.
5. Mỗi item phải gắn source_indices (chỉ số 0-indexed) tham chiếu đến \
contexts đầu vào.
6. Nếu không đủ căn cứ cho một khía cạnh, đặt confidence = "insufficient_evidence".

CÁC KHÍA CẠNH CẦN ĐÁNH GIÁ (aspect):
- "Mục tiêu đào tạo": Mức độ phù hợp của mục tiêu chung và mục tiêu cụ thể \
với nhu cầu xã hội, thị trường lao động, sứ mạng của trường.
- "Chuẩn đầu ra": Mức độ phù hợp, đo lường được, và đáp ứng yêu cầu \
kiến thức, kỹ năng, năng lực tự chủ và trách nhiệm.
- "Nội dung chương trình": Sự thống nhất giữa mục tiêu, CĐR, và nội dung \
chương trình; tính cập nhật và phù hợp thực tiễn.
- "Cấu trúc học phần": Tính hợp lý của danh sách học phần, mã học phần, \
phân bổ lý thuyết/thực hành, tiên quyết, logic trình tự học kỳ.
- "Ma trận mục tiêu - CĐR": Mức độ liên kết giữa mục tiêu đào tạo và \
chuẩn đầu ra; có phủ đầy đủ không.
- "Ma trận học phần - CĐR": Mức độ đóng góp của từng học phần vào CĐR; \
có học phần nào không đóng góp hoặc CĐR nào thiếu học phần không.
- "Khối lượng tín chỉ": Tổng tín chỉ, phân bổ giữa đại cương/chuyên ngành/\
tự chọn; so với quy định (TT17, khung trình độ).
- "Điều kiện đảm bảo": Đội ngũ giảng viên, cơ sở vật chất, tài liệu, \
phương pháp giảng dạy và đánh giá.
- "Các bên liên quan": Mức độ tham gia của sinh viên, cựu sinh viên, \
nhà tuyển dụng, giảng viên vào quá trình xây dựng và cải tiến CTĐT.

OUTPUT FORMAT (JSON):
{
  "items": [
    {
      "aspect": "Mục tiêu đào tạo",
      "finding": "Phát hiện cụ thể từ tài liệu (2-4 câu)",
      "recommendation": "Đề xuất hướng cải tiến ở mức nháp (1-3 câu)",
      "rationale": "Căn cứ từ tài liệu cho đánh giá này",
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


class CurrentCurriculumReviewSkill:
    """
    AI skill for evaluating the current curriculum effectiveness.

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
    ) -> CurrentCurriculumReviewResult:
        """
        Run curriculum review on the provided sources.

        Returns:
          - insufficient_evidence if no sources
          - needs_generation if LLM is disabled
          - generated if LLM succeeds
          - failed if LLM errors
        """
        # ── No contexts → insufficient_evidence ──────────────────
        if not sources:
            return CurrentCurriculumReviewResult(
                status=ReviewStatus.INSUFFICIENT_EVIDENCE.value,
                items=[],
                warnings=["Không có tài liệu CTĐT nào được tìm thấy trong phạm vi đợt cập nhật."],
            )

        # ── LLM disabled → deterministic fallback ────────────────
        if not getattr(settings, "SYNTHESIS_ENABLED", False):
            return CurrentCurriculumReviewResult(
                status=ReviewStatus.NEEDS_GENERATION.value,
                items=[],
                warnings=[
                    "SYNTHESIS_ENABLED=false. Cần bật LLM để sinh đánh giá CTĐT.",
                    f"Có {len(sources)} nguồn tài liệu sẵn sàng để đánh giá.",
                ],
            )

        # ── Check OpenAI key ─────────────────────────────────────
        api_key = getattr(settings, "OPENAI_API_KEY", None)
        if not api_key:
            return CurrentCurriculumReviewResult(
                status=ReviewStatus.NEEDS_GENERATION.value,
                items=[],
                warnings=[
                    "OPENAI_API_KEY chưa được cấu hình.",
                    f"Có {len(sources)} nguồn tài liệu sẵn sàng để đánh giá.",
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
                "curriculum_review_skill.llm_failed update_cycle=%s error=%s",
                update_cycle_id, exc.__class__.__name__,
            )
            return CurrentCurriculumReviewResult(
                status=ReviewStatus.FAILED.value,
                items=[],
                warnings=[
                    f"LLM gọi thất bại: {exc.__class__.__name__}",
                    f"Có {len(sources)} nguồn tài liệu nhưng chưa thể đánh giá.",
                ],
            )

        # ── Parse response ───────────────────────────────────────
        try:
            result = self._parse_response(raw_json, sources)
        except Exception as exc:
            logger.warning(
                "curriculum_review_skill.parse_failed update_cycle=%s error=%s",
                update_cycle_id, str(exc)[:200],
            )
            return CurrentCurriculumReviewResult(
                status=ReviewStatus.FAILED.value,
                items=[],
                warnings=[
                    f"Lỗi parse LLM response: {str(exc)[:150]}",
                ],
            )

        logger.info(
            "curriculum_review_skill.done update_cycle=%s items=%d status=%s",
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
    ) -> CurrentCurriculumReviewResult:
        """Parse LLM JSON response and validate source references."""
        data = json.loads(raw_json)

        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")

        raw_items = data.get("items", [])
        warnings = list(data.get("warnings", []))

        if not isinstance(raw_items, list):
            raise ValueError("'items' is not a list")

        items: list[CurriculumEvaluationPoint] = []

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue

            # Validate confidence
            confidence = raw.get("confidence", "medium")
            try:
                ReviewConfidence(confidence)
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

            items.append(CurriculumEvaluationPoint(
                aspect=str(raw.get("aspect", "")).strip() or "Đánh giá chung",
                finding=str(raw.get("finding", "")).strip(),
                recommendation=str(raw.get("recommendation", "")).strip(),
                rationale=str(raw.get("rationale", "")).strip(),
                confidence=confidence,
                sources=resolved_sources,
            ))

        status = (
            ReviewStatus.GENERATED.value
            if items
            else ReviewStatus.INSUFFICIENT_EVIDENCE.value
        )

        return CurrentCurriculumReviewResult(
            status=status,
            items=items,
            warnings=warnings,
        )
