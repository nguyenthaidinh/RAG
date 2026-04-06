"""
CTDT AI Server — Endpoint chuyên biệt cho module "Cập nhật Chương trình Đào tạo".

Tích hợp với Moodle LMS để:
  - Trả lời câu hỏi về chương trình đào tạo dựa trên tài liệu đã ingest
  - Đề xuất nội dung cập nhật khi nhận yêu cầu từ Moodle
  - Kiểm tra tính nhất quán giữa tài liệu nội bộ và chuẩn chương trình

Routes:
  POST /api/v1/ctdt/query       — Truy vấn thông tin chương trình đào tạo
  POST /api/v1/ctdt/review      — Nhận yêu cầu rà soát từ Moodle, trả về đề xuất
  GET  /api/v1/ctdt/health      — Kiểm tra trạng thái endpoint (no-auth)
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import get_current_user
from app.core.config import settings
from app.db.models.user import User
from app.db.session import get_db
from app.services.retrieval.factories import get_query_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ctdt", tags=["ctdt"])


# ──────────────────────────────────────────────────────────────────────────────
# Shared lazy singleton
# ──────────────────────────────────────────────────────────────────────────────

_query_svc = None


def _get_query_svc():
    global _query_svc
    if _query_svc is None:
        _query_svc = get_query_service()
    return _query_svc


# ──────────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────────


class MoodleCourseContext(BaseModel):
    """Thông tin ngữ cảnh khoá học từ Moodle (tùy chọn)."""
    course_id: int | None = None
    course_fullname: str | None = None
    course_shortname: str | None = None
    category: str | None = None


class CTDTQueryRequest(BaseModel):
    """
    Yêu cầu truy vấn thông tin chương trình đào tạo.

    - question: câu hỏi về chương trình đào tạo
    - context:  ngữ cảnh khoá học từ Moodle (giúp giới hạn phạm vi tìm kiếm)
    - mode:     chiến lược retrieval (hybrid được khuyến nghị)
    - final_limit: số chunk trả về tối đa
    """
    question: str = Field(min_length=1, max_length=4000, description="Câu hỏi về chương trình đào tạo")
    context: MoodleCourseContext | None = None
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    final_limit: int = Field(default=10, ge=1, le=50)


class CTDTSourceChunk(BaseModel):
    document_id: int
    chunk_id: int
    score: float
    snippet: str


class CTDTQueryResponse(BaseModel):
    answer: str | None = None
    sources: list[CTDTSourceChunk]
    count: int
    latency_ms: int


class CTDTReviewRequest(BaseModel):
    """
    Yêu cầu rà soát / cập nhật chương trình đào tạo từ Moodle.

    Moodle gửi nội dung hiện tại của một học phần, AI Server phân tích
    và trả về các đề xuất cập nhật dựa trên tài liệu chuẩn đã được ingest.
    """
    course_id: int = Field(description="Moodle course ID")
    section_title: str = Field(min_length=1, max_length=500)
    current_content: str = Field(min_length=1, max_length=8000, description="Nội dung hiện tại cần rà soát")
    review_focus: str | None = Field(
        default=None,
        max_length=1000,
        description="Hướng dẫn rà soát cụ thể (ví dụ: 'kiểm tra chuẩn đầu ra', 'so sánh với chương trình khung')",
    )


class CTDTReviewSuggestion(BaseModel):
    type: Literal["addition", "removal", "update", "note"]
    description: str
    reference_chunk_id: int | None = None
    reference_document_id: int | None = None


class CTDTReviewResponse(BaseModel):
    course_id: int
    section_title: str
    summary: str | None = None
    suggestions: list[CTDTReviewSuggestion]
    sources: list[CTDTSourceChunk]
    latency_ms: int


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/health")
async def ctdt_health():
    """Kiểm tra trạng thái CTDT endpoint (no-auth)."""
    return {"status": "ok", "service": "ctdt"}


@router.post("/query", response_model=CTDTQueryResponse)
async def ctdt_query(
    body: CTDTQueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Truy vấn thông tin chương trình đào tạo.

    Nhận câu hỏi tự nhiên về CTĐT, trả về câu trả lời tổng hợp
    kèm nguồn tham chiếu từ tài liệu nội bộ đã được ingest.
    """
    t0 = time.monotonic()
    tenant_id = user.tenant_id
    user_id = user.id

    # Nếu có ngữ cảnh Moodle, bổ sung vào câu hỏi để cải thiện độ chính xác
    enriched_question = body.question
    if body.context and body.context.course_fullname:
        enriched_question = (
            f"[Khoá học: {body.context.course_fullname}] {body.question}"
        )

    try:
        query_svc = _get_query_svc()
        results = await query_svc.query(
            tenant_id=tenant_id,
            user_id=user_id,
            query_text=enriched_question,
            idempotency_key=None,
            final_limit=body.final_limit,
            mode=body.mode,
            include_debug=False,
            history=[],
        )
    except Exception as exc:
        logger.error("ctdt.query_failed tenant_id=%s: %s", tenant_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi xử lý truy vấn chương trình đào tạo.")

    # ── Answer synthesis (best-effort) ───────────────────────────────────────
    answer: str | None = None
    try:
        from app.services.answer_service import AnswerService, AnswerSnippet
        snippets = [
            AnswerSnippet(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                snippet=r.snippet,
                score=r.score,
            )
            for r in results[:settings.LLM_ANSWER_MAX_RESULTS]
        ]
        answer = await AnswerService().generate(
            question=body.question,
            snippets=snippets,
            history=[],
        )
    except Exception:
        answer = None

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return CTDTQueryResponse(
        answer=answer,
        sources=[
            CTDTSourceChunk(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                score=r.score,
                snippet=r.snippet,
            )
            for r in results
        ],
        count=len(results),
        latency_ms=elapsed_ms,
    )


@router.post("/review", response_model=CTDTReviewResponse)
async def ctdt_review(
    body: CTDTReviewRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Rà soát nội dung học phần từ Moodle và đề xuất cập nhật.

    Moodle gửi nội dung một học phần, AI Server:
    1. Tìm tài liệu liên quan trong kho nội bộ
    2. So sánh nội dung hiện tại với chuẩn
    3. Trả về danh sách đề xuất cập nhật có trích dẫn nguồn
    """
    t0 = time.monotonic()
    tenant_id = user.tenant_id
    user_id = user.id

    focus = body.review_focus or "rà soát tính đầy đủ và cập nhật so với chương trình chuẩn"
    review_query = (
        f"Rà soát học phần '{body.section_title}': {focus}\n\n"
        f"Nội dung hiện tại:\n{body.current_content[:3000]}"
    )

    try:
        query_svc = _get_query_svc()
        results = await query_svc.query(
            tenant_id=tenant_id,
            user_id=user_id,
            query_text=review_query,
            idempotency_key=None,
            final_limit=10,
            mode="hybrid",
            include_debug=False,
            history=[],
        )
    except Exception as exc:
        logger.error("ctdt.review_failed tenant_id=%s: %s", tenant_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Lỗi xử lý rà soát chương trình đào tạo.")

    # ── LLM-powered review synthesis ─────────────────────────────────────────
    summary: str | None = None
    suggestions: list[CTDTReviewSuggestion] = []

    try:
        from app.services.answer_service import AnswerService, AnswerSnippet

        snippets = [
            AnswerSnippet(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                snippet=r.snippet,
                score=r.score,
            )
            for r in results[:settings.LLM_ANSWER_MAX_RESULTS]
        ]

        synthesis_prompt = (
            f"Bạn là chuyên gia rà soát chương trình đào tạo đại học.\n"
            f"Dựa trên các tài liệu tham chiếu dưới đây, hãy rà soát nội dung học phần "
            f"'{body.section_title}' và đưa ra nhận xét ngắn gọn về tính đầy đủ, "
            f"cập nhật và phù hợp với chuẩn chương trình đào tạo.\n"
            f"Hướng dẫn rà soát: {focus}"
        )

        summary = await AnswerService().generate(
            question=synthesis_prompt,
            snippets=snippets,
            history=[],
        )

        # Tạo đề xuất cơ bản từ chunks liên quan
        for r in results[:5]:
            suggestions.append(
                CTDTReviewSuggestion(
                    type="note",
                    description=f"Tham chiếu tài liệu (score={r.score:.2f}): {r.snippet[:200]}...",
                    reference_chunk_id=r.chunk_id,
                    reference_document_id=r.document_id,
                )
            )

    except Exception:
        logger.warning("ctdt.review_synthesis_failed", exc_info=True)

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return CTDTReviewResponse(
        course_id=body.course_id,
        section_title=body.section_title,
        summary=summary,
        suggestions=suggestions,
        sources=[
            CTDTSourceChunk(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                score=r.score,
                snippet=r.snippet,
            )
            for r in results
        ],
        latency_ms=elapsed_ms,
    )
