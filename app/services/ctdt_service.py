"""
CTDT Service — Logic RAG + LLM cho các tác vụ chuyên biệt CTĐT.

Cung cấp 3 tác vụ:
  - suggest_objectives: Đề xuất mục tiêu đào tạo
  - suggest_outcomes:   Đề xuất chuẩn đầu ra
  - suggest_mapping:    Tạo ma trận liên kết CĐR - học phần
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.core.config import settings


@dataclass(frozen=True)
class RetrievalConfidence:
    """Confidence signal for RAG retrieval quality."""
    level: str  # "high" | "medium" | "low" | "no_reference"
    retrieved_count: int
    top_score: float | None
    reason: str


def _compute_confidence(
    retrieved_count: int,
    top_score: float | None,
    failure_reason: str | None,
) -> RetrievalConfidence:
    """Compute retrieval confidence from count + score.

    Evaluated top-down: no_reference → high → medium → low (default).
    """
    if failure_reason is not None or retrieved_count == 0:
        return RetrievalConfidence(
            level="no_reference",
            retrieved_count=retrieved_count,
            top_score=top_score,
            reason=failure_reason or "no_results",
        )

    score = top_score if top_score is not None else 0.0

    if retrieved_count >= 3 and score >= 0.6:
        return RetrievalConfidence(
            level="high",
            retrieved_count=retrieved_count,
            top_score=top_score,
            reason="sufficient_references",
        )

    if retrieved_count >= 2 and score >= 0.3:
        return RetrievalConfidence(
            level="medium",
            retrieved_count=retrieved_count,
            top_score=top_score,
            reason="partial_references",
        )

    return RetrievalConfidence(
        level="low",
        retrieved_count=retrieved_count,
        top_score=top_score,
        reason="few_or_weak_references",
    )

logger = logging.getLogger(__name__)

# ── Prompt templates (hardcoded — không cho user tùy ý) ──────────────────────

_OBJECTIVES_PROMPT = """\
Bạn là chuyên gia thiết kế chương trình đào tạo đại học Việt Nam, tuân thủ Thông tư 17/2021/TT-BGDĐT.

Ngành: {program_name}
Trình độ: {degree_level}

TÀI LIỆU THAM CHIẾU (các CTĐT mẫu từ knowledge base):
{reference_context}

NỘI DUNG HIỆN TẠI (nếu có):
{current_content}

Hãy đề xuất MỤC TIÊU ĐÀO TẠO cho chương trình này. Trả về JSON array theo định dạng:
[
  {{
    "code": "MT1",
    "type": "general|specific",
    "content": "Nội dung mục tiêu (tiếng Việt, rõ ràng, đo lường được)",
    "rationale": "Lý do đề xuất / cơ sở pháp lý"
  }}
]

Quy tắc:
- Mục tiêu chung (type=general): 1-2 mục tiêu, phản ánh định hướng nghề nghiệp tổng quát.
- Mục tiêu cụ thể (type=specific): 4-6 mục tiêu, mỗi mục tiêu gắn với một nhóm năng lực.
- Tuân thủ Điều 4, TT17/2021/TT-BGDĐT.
- Chỉ trả về JSON, không giải thích thêm."""

_OUTCOMES_PROMPT = """\
Bạn là chuyên gia thiết kế chuẩn đầu ra đại học Việt Nam, tuân thủ Thông tư 17/2021/TT-BGDĐT và khung trình độ quốc gia.

Ngành: {program_name}
Trình độ: {degree_level}

MỤC TIÊU ĐÃ DUYỆT:
{objectives_json}

TÀI LIỆU THAM CHIẾU (CĐR mẫu từ knowledge base):
{reference_context}

Hãy đề xuất CHUẨN ĐẦU RA cho chương trình này. Trả về JSON array theo định dạng:
[
  {{
    "code": "CĐR1",
    "category": "knowledge|skill|autonomy",
    "content": "Nội dung chuẩn đầu ra (động từ hành động, đo lường được)",
    "bloom_level": "remember|understand|apply|analyze|evaluate|create",
    "maps_to_objectives": ["MT1", "MT2"]
  }}
]

Quy tắc:
- knowledge: kiến thức chuyên môn và khoa học cơ bản.
- skill: kỹ năng nghề nghiệp, giao tiếp, làm việc nhóm, CNTT.
- autonomy: mức tự chủ, trách nhiệm nghề nghiệp.
- Mỗi CĐR liên kết ít nhất 1 mục tiêu trong maps_to_objectives.
- Chỉ trả về JSON, không giải thích thêm."""

_REVIEW_PROMPT = """\
Bạn là chuyên gia rà soát chương trình đào tạo đại học Việt Nam, tuân thủ Thông tư 17/2021/TT-BGDĐT.

HỌC PHẦN: {section_title}
HƯỚNG RÀ SOÁT: {focus}

NỘI DUNG HIỆN TẠI:
{current_content}

TÀI LIỆU THAM CHIẾU (từ knowledge base):
{reference_context}

Hãy rà soát nội dung học phần và trả về JSON theo định dạng:
{{
  "summary": "Nhận xét tổng quan ngắn gọn (2-3 câu, tiếng Việt)",
  "suggestions": [
    {{
      "type": "addition|removal|update|note",
      "description": "Mô tả cụ thể đề xuất (tiếng Việt)"
    }}
  ]
}}

Quy tắc:
- addition: đề xuất bổ sung nội dung còn thiếu.
- removal: đề xuất loại bỏ nội dung lỗi thời hoặc không phù hợp.
- update: đề xuất cập nhật / chỉnh sửa nội dung hiện có.
- note: ghi chú tham khảo hoặc lưu ý chung.
- Chỉ trả về JSON, không giải thích thêm."""

_MAPPING_PROMPT = """\
Bạn là chuyên gia xây dựng ma trận liên kết chương trình đào tạo.

CHUẨN ĐẦU RA:
{outcomes_json}

HỌC PHẦN:
{courses_json}

TÀI LIỆU THAM CHIẾU (ma trận mẫu từ knowledge base):
{reference_context}

Hãy tạo MA TRẬN LIÊN KẾT giữa chuẩn đầu ra và học phần. Trả về JSON theo định dạng:
{{
  "matrix": [
    {{
      "outcome_code": "CĐR1",
      "course_code": "MH001",
      "contribution": "high|medium|low"
    }}
  ],
  "coverage_gaps": ["CĐR không được hỗ trợ bởi học phần nào"]
}}

Quy tắc:
- high: học phần là phương tiện chính phát triển CĐR này.
- medium: học phần đóng góp đáng kể nhưng không phải chính.
- low: học phần có đóng góp nhỏ.
- Liệt kê coverage_gaps nếu có CĐR không được bất kỳ học phần nào hỗ trợ.
- Chỉ trả về JSON, không giải thích thêm."""


async def _retrieve_context(
    *,
    query_svc: Any,
    tenant_id: int,
    user_id: int,
    query_text: str,
    fn_name: str,
    fetch_limit: int = 8,
    context_limit: int = 5,
) -> tuple[str, int, str | None, RetrievalConfidence]:
    """
    Retrieve reference context từ knowledge base.

    Returns:
        (reference_context, retrieved_count, failure_reason, confidence)
    """
    try:
        results = await query_svc.query(
            tenant_id=tenant_id,
            user_id=user_id,
            query_text=query_text,
            idempotency_key=None,
            final_limit=fetch_limit,
            mode="hybrid",
            include_debug=False,
            history=[],
        )
        top = results[:context_limit]
        parts = [f"[{i}] doc={r.document_id}\n{r.snippet[:600]}" for i, r in enumerate(top, start=1)]
        context = "\n\n".join(parts)

        top_score = getattr(top[0], "score", None) if top else None

        if not context:
            logger.warning(
                "%s.no_reference_context reason=no_results query=%r",
                fn_name, query_text,
            )
            confidence = _compute_confidence(0, None, "no_results")
            return "", 0, "no_results", confidence

        confidence = _compute_confidence(len(top), top_score, None)
        return context, len(top), None, confidence

    except (ConnectionError, asyncio.TimeoutError) as exc:
        logger.warning(
            "%s.retrieve_failed reason=connection_error error_type=%s error=%s",
            fn_name, type(exc).__name__, exc,
        )
        logger.warning("%s.no_reference_context reason=connection_error", fn_name)
        confidence = _compute_confidence(0, None, "connection_error")
        return "", 0, "connection_error", confidence

    except Exception as exc:
        logger.warning(
            "%s.retrieve_failed reason=retrieve_error error_type=%s error=%s",
            fn_name, type(exc).__name__, exc,
        )
        logger.warning("%s.no_reference_context reason=retrieve_error", fn_name)
        confidence = _compute_confidence(0, None, "retrieve_error")
        return "", 0, "retrieve_error", confidence


def _extract_json(text: str) -> Any:
    """Parse JSON từ LLM response, hỗ trợ markdown code blocks."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return json.loads(text.strip())


async def _llm_call(prompt: str, model: str | None = None, timeout_s: float = 20.0) -> str | None:
    """Gọi OpenAI API và trả về text. Fail-open: trả None nếu lỗi."""
    api_key = getattr(settings, "OPENAI_API_KEY", None)
    if not api_key:
        return None

    model = model or getattr(settings, "LLM_ANSWER_MODEL", "gpt-4o-mini")
    temperature = float(getattr(settings, "LLM_ANSWER_TEMPERATURE", 0.2))

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}],
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s + 1.0) as client:
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
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        token_usage = data.get("usage", {})

        logger.info(
            "ctdt_service.llm_ok model=%s latency_ms=%d prompt_tokens=%s completion_tokens=%s",
            model,
            elapsed_ms,
            token_usage.get("prompt_tokens"),
            token_usage.get("completion_tokens"),
        )
        return text.strip() if isinstance(text, str) else None

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(
            "ctdt_service.llm_failed model=%s latency_ms=%d error=%s",
            model,
            elapsed_ms,
            exc.__class__.__name__,
        )
        return None


async def suggest_objectives(
    *,
    query_svc: Any,
    tenant_id: int,
    user_id: int,
    program_name: str,
    degree_level: str,
    current_content: str = "",
) -> dict:
    """
    Đề xuất mục tiêu đào tạo dựa trên RAG + LLM.

    Returns: {"items": list[dict], "retrieved_count": int}
    """
    reference_context, retrieved_count, _, confidence = await _retrieve_context(
        query_svc=query_svc,
        tenant_id=tenant_id,
        user_id=user_id,
        query_text=f"mục tiêu đào tạo ngành {program_name} {degree_level}",
        fn_name="ctdt_service.suggest_objectives",
    )

    prompt = _OBJECTIVES_PROMPT.format(
        program_name=program_name,
        degree_level=degree_level,
        reference_context=reference_context or "(không có tài liệu tham chiếu)",
        current_content=current_content[:2000] if current_content else "(không có)",
    )

    raw = await _llm_call(prompt)
    if not raw:
        return {"items": [], "retrieved_count": retrieved_count, "confidence": confidence}

    try:
        data = _extract_json(raw)
        if isinstance(data, list):
            return {"items": data, "retrieved_count": retrieved_count, "confidence": confidence}
    except Exception as exc:
        logger.warning("ctdt_service.suggest_objectives.parse_failed: %s | raw=%s", exc, raw[:200])

    return {"items": [], "retrieved_count": retrieved_count, "confidence": confidence}


async def suggest_outcomes(
    *,
    query_svc: Any,
    tenant_id: int,
    user_id: int,
    program_name: str,
    degree_level: str,
    objectives: list[dict],
) -> dict:
    """
    Đề xuất chuẩn đầu ra dựa trên mục tiêu đã duyệt + RAG + LLM.

    Returns: {"items": list[dict], "retrieved_count": int}
    """
    reference_context, retrieved_count, _, confidence = await _retrieve_context(
        query_svc=query_svc,
        tenant_id=tenant_id,
        user_id=user_id,
        query_text=f"chuẩn đầu ra ngành {program_name} {degree_level}",
        fn_name="ctdt_service.suggest_outcomes",
    )

    objectives_json = json.dumps(objectives, ensure_ascii=False, indent=2)
    prompt = _OUTCOMES_PROMPT.format(
        program_name=program_name,
        degree_level=degree_level,
        objectives_json=objectives_json,
        reference_context=reference_context or "(không có tài liệu tham chiếu)",
    )

    raw = await _llm_call(prompt)
    if not raw:
        return {"items": [], "retrieved_count": retrieved_count, "confidence": confidence}

    try:
        data = _extract_json(raw)
        if isinstance(data, list):
            return {"items": data, "retrieved_count": retrieved_count, "confidence": confidence}
    except Exception as exc:
        logger.warning("ctdt_service.suggest_outcomes.parse_failed: %s | raw=%s", exc, raw[:200])

    return {"items": [], "retrieved_count": retrieved_count, "confidence": confidence}


async def review_section(
    *,
    section_title: str,
    current_content: str,
    focus: str,
    results: list,
) -> dict:
    """
    Rà soát nội dung học phần dựa trên retrieval results đã có.

    Returns: {"summary": str | None, "suggestions": [{"type": ..., "description": ...}]}
    """
    parts = []
    for i, r in enumerate(results[:5], start=1):
        parts.append(f"[{i}] doc={r.document_id}\n{r.snippet[:600]}")
    reference_context = "\n\n".join(parts) or "(không có tài liệu tham chiếu)"

    prompt = _REVIEW_PROMPT.format(
        section_title=section_title,
        focus=focus,
        current_content=current_content[:2000],
        reference_context=reference_context,
    )

    raw = await _llm_call(prompt)
    if not raw:
        return {"summary": None, "suggestions": []}

    try:
        data = _extract_json(raw)
        if isinstance(data, dict):
            return {
                "summary": data.get("summary"),
                "suggestions": data.get("suggestions", []),
            }
    except Exception as exc:
        logger.warning("ctdt_service.review_section.parse_failed: %s | raw=%s", exc, raw[:200])

    return {"summary": None, "suggestions": []}


async def suggest_mapping(
    *,
    query_svc: Any,
    tenant_id: int,
    user_id: int,
    program_name: str,
    outcomes: list[dict],
    courses: list[dict],
) -> dict:
    """
    Tạo ma trận liên kết CĐR - học phần bằng RAG + LLM.

    Returns: dict với keys 'matrix' và 'coverage_gaps'.
    """
    # Retrieve ma trận mẫu từ knowledge base
    reference_context, retrieved_count, _, confidence = await _retrieve_context(
        query_svc=query_svc,
        tenant_id=tenant_id,
        user_id=user_id,
        query_text=f"ma trận liên kết chuẩn đầu ra học phần {program_name}",
        fn_name="ctdt_service.suggest_mapping",
    )

    outcomes_json = json.dumps(outcomes, ensure_ascii=False, indent=2)
    courses_json = json.dumps(courses, ensure_ascii=False, indent=2)

    prompt = _MAPPING_PROMPT.format(
        outcomes_json=outcomes_json,
        courses_json=courses_json,
        reference_context=reference_context or "(không có tài liệu tham chiếu)",
    )

    raw = await _llm_call(prompt)
    if not raw:
        return {"matrix": [], "coverage_gaps": [], "confidence": confidence}

    try:
        data = _extract_json(raw)
        if isinstance(data, dict):
            return {
                "matrix": data.get("matrix", []),
                "coverage_gaps": data.get("coverage_gaps", []),
                "confidence": confidence,
            }
    except Exception as exc:
        logger.warning("ctdt_service.suggest_mapping.parse_failed: %s | raw=%s", exc, raw[:200])

    return {"matrix": [], "coverage_gaps": [], "confidence": confidence}
