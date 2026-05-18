"""
ChangeProposalSkill — Dự thảo nội dung cần cập nhật CTĐT phục vụ Mẫu 06.

Mẫu 06: "Dự thảo những nội dung cần cập nhật, bổ sung, thay đổi,
cải tiến chương trình đào tạo."

R5.4:
- Nhận contexts/sources từ R4 change_proposal retrieval task.
- Nếu SYNTHESIS_ENABLED=true → gọi OpenAI để dự thảo đề xuất thay đổi.
- Nếu SYNTHESIS_ENABLED=false → trả deterministic fallback.
- Không bịa nội dung, không tự tạo nguồn giả.
- Không ghi vào CTĐT chính thức, không sinh toàn bộ mục tiêu/CĐR/học phần.

Architecture::

    run()
        ├─ no contexts? → insufficient_evidence
        ├─ LLM disabled? → needs_generation (deterministic)
        ├─ build prompt (Mẫu 06 specific)
        ├─ call OpenAI
        ├─ parse JSON response → ChangeProposalResult
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


class ChangeType(str, Enum):
    ADD = "add"
    UPDATE = "update"
    REMOVE = "remove"
    RESTRUCTURE = "restructure"
    CLARIFY = "clarify"
    KEEP_WITH_MONITORING = "keep_with_monitoring"
    OTHER = "other"


class ProposalPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProposalConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ProposalStatus(str, Enum):
    GENERATED = "generated"
    NEEDS_GENERATION = "needs_generation"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


# ── Result DTOs ──────────────────────────────────────────────────────


@dataclass
class ChangeProposalItem:
    """One change proposal for the curriculum update."""
    target_area: str
    change_type: str  # ChangeType value
    current_issue: str
    proposed_change: str
    rationale: str
    expected_impact: str
    priority: str  # ProposalPriority value
    confidence: str  # ProposalConfidence value
    sources: list[AnalysisSource]


@dataclass
class ChangeProposalResult:
    """Full result from change proposal skill."""
    status: str  # ProposalStatus value
    items: list[ChangeProposalItem]
    warnings: list[str]
    task_type: str = "change_proposal"


# ── Prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Bạn là chuyên gia tư vấn cập nhật chương trình đào tạo đại học Việt Nam.

NHIỆM VỤ: Dự thảo những nội dung cần cập nhật, bổ sung, thay đổi, cải tiến \
chương trình đào tạo (CTĐT), dựa trên các đoạn trích tài liệu được cung cấp \
(phục vụ Mẫu 06).

QUY TẮC BẮT BUỘC:
1. CHỈ đề xuất dựa trên các đoạn trích (contexts) được cung cấp.
2. KHÔNG bịa thêm thông tin, số liệu, mã học phần, số tín chỉ, mã CĐR, \
hoặc nguồn không có trong contexts.
3. KHÔNG tự ghi vào CTĐT chính thức. Đây chỉ là DỰ THẢO đề xuất.
4. KHÔNG sinh toàn bộ danh sách mục tiêu, chuẩn đầu ra, hoặc học phần mới.
5. KHÔNG xuất file.
6. Mỗi item phải gắn source_indices (chỉ số 0-indexed) tham chiếu đến \
contexts đầu vào.
7. Nếu không đủ căn cứ, đặt confidence = "insufficient_evidence".

CÁC LĨNH VỰC CẦN XEM XÉT (target_area):
- "Mục tiêu đào tạo": Cập nhật mục tiêu chung/cụ thể
- "Chuẩn đầu ra": Bổ sung, chỉnh sửa CĐR
- "Nội dung chương trình": Cấu trúc chương trình tổng thể
- "Cấu trúc học phần": Thay đổi cấu trúc, phân bổ học phần
- "Học phần": Thêm/bớt/sửa học phần cụ thể
- "Ma trận mục tiêu - CĐR": Điều chỉnh liên kết mục tiêu-CĐR
- "Ma trận học phần - CĐR": Điều chỉnh đóng góp học phần-CĐR
- "Phương pháp đánh giá": Cải tiến phương pháp kiểm tra đánh giá
- "Điều kiện đảm bảo": Đội ngũ, CSVC, tài liệu
- "Minh chứng / quy trình": Bổ sung minh chứng, cải tiến quy trình
- "Khác": Nội dung khác

LOẠI THAY ĐỔI (change_type):
- add: Thêm mới
- update: Cập nhật/chỉnh sửa nội dung hiện có
- remove: Loại bỏ nội dung không còn phù hợp
- restructure: Tái cấu trúc/sắp xếp lại
- clarify: Làm rõ/chi tiết hóa nội dung
- keep_with_monitoring: Giữ nguyên nhưng cần theo dõi
- other: Khác

OUTPUT FORMAT (JSON):
{
  "items": [
    {
      "target_area": "Chuẩn đầu ra",
      "change_type": "update|add|remove|restructure|clarify|keep_with_monitoring|other",
      "current_issue": "Vấn đề hiện tại cần thay đổi (2-3 câu)",
      "proposed_change": "Nội dung đề xuất thay đổi cụ thể (2-4 câu)",
      "rationale": "Lý do/căn cứ cho thay đổi này",
      "expected_impact": "Tác động dự kiến của thay đổi",
      "priority": "high|medium|low",
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


class ChangeProposalSkill:
    """
    AI skill for drafting change proposals for curriculum update.

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
    ) -> ChangeProposalResult:
        """
        Run change proposal drafting on the provided sources.

        Returns:
          - insufficient_evidence if no sources
          - needs_generation if LLM is disabled
          - generated if LLM succeeds
          - failed if LLM errors
        """
        # ── No contexts → insufficient_evidence ──────────────────
        if not sources:
            return ChangeProposalResult(
                status=ProposalStatus.INSUFFICIENT_EVIDENCE.value,
                items=[],
                warnings=["Không có tài liệu nào được tìm thấy trong phạm vi đợt cập nhật."],
            )

        # ── LLM disabled → deterministic fallback ────────────────
        if not getattr(settings, "SYNTHESIS_ENABLED", False):
            return ChangeProposalResult(
                status=ProposalStatus.NEEDS_GENERATION.value,
                items=[],
                warnings=[
                    "SYNTHESIS_ENABLED=false. Cần bật LLM để sinh dự thảo đề xuất.",
                    f"Có {len(sources)} nguồn tài liệu sẵn sàng để phân tích.",
                ],
            )

        # ── Check OpenAI key ─────────────────────────────────────
        api_key = getattr(settings, "OPENAI_API_KEY", None)
        if not api_key:
            return ChangeProposalResult(
                status=ProposalStatus.NEEDS_GENERATION.value,
                items=[],
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
                "change_proposal_skill.llm_failed update_cycle=%s error=%s",
                update_cycle_id, exc.__class__.__name__,
            )
            return ChangeProposalResult(
                status=ProposalStatus.FAILED.value,
                items=[],
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
                "change_proposal_skill.parse_failed update_cycle=%s error=%s",
                update_cycle_id, str(exc)[:200],
            )
            return ChangeProposalResult(
                status=ProposalStatus.FAILED.value,
                items=[],
                warnings=[
                    f"Lỗi parse LLM response: {str(exc)[:150]}",
                ],
            )

        logger.info(
            "change_proposal_skill.done update_cycle=%s items=%d status=%s",
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
    ) -> ChangeProposalResult:
        """Parse LLM JSON response and validate source references."""
        data = json.loads(raw_json)

        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")

        raw_items = data.get("items", [])
        warnings = list(data.get("warnings", []))

        if not isinstance(raw_items, list):
            raise ValueError("'items' is not a list")

        items: list[ChangeProposalItem] = []

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue

            # Validate change_type
            change_type = raw.get("change_type", "other")
            try:
                ChangeType(change_type)
            except ValueError:
                change_type = "other"

            # Validate priority
            priority = raw.get("priority", "medium")
            try:
                ProposalPriority(priority)
            except ValueError:
                priority = "medium"

            # Validate confidence
            confidence = raw.get("confidence", "medium")
            try:
                ProposalConfidence(confidence)
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

            items.append(ChangeProposalItem(
                target_area=str(raw.get("target_area", "")).strip() or "Khác",
                change_type=change_type,
                current_issue=str(raw.get("current_issue", "")).strip(),
                proposed_change=str(raw.get("proposed_change", "")).strip(),
                rationale=str(raw.get("rationale", "")).strip(),
                expected_impact=str(raw.get("expected_impact", "")).strip(),
                priority=priority,
                confidence=confidence,
                sources=resolved_sources,
            ))

        status = (
            ProposalStatus.GENERATED.value
            if items
            else ProposalStatus.INSUFFICIENT_EVIDENCE.value
        )

        return ChangeProposalResult(
            status=status,
            items=items,
            warnings=warnings,
        )
