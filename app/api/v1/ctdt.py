"""
CTĐT AI Processing Engine — Endpoints cho hệ thống xử lý AI phục vụ
quy trình Cập nhật Chương trình Đào tạo (CTĐT).

Engine cung cấp:
  - Document ingestion pipeline (download → extract → chunk → embed → index)
  - Metadata-scoped retrieval (update_cycle / program / document_role)
  - LLM-powered suggestion & review (objectives, outcomes, mapping)

─── New CTĐT Engine endpoints (R1–R6) ───────────────────────────────
  POST /api/v1/ctdt/documents/ingest   — Ingest file từ FileServer URL + metadata nghiệp vụ
  GET  /api/v1/ctdt/documents/{id}     — Trạng thái tài liệu đã ingest
  POST /api/v1/ctdt/retrieve           — Truy xuất tài liệu theo phạm vi nghiệp vụ
  GET  /api/v1/ctdt/health             — Health check (no-auth)
  POST /api/v1/ctdt/update-cycles/analyze — Phân tích đợt cập nhật CTĐT (R4 skeleton / R5 draft)
  GET  /update-cycles/{id}/analysis-drafts/latest — Lấy draft phân tích mới nhất
  GET  /update-cycles/{id}/mau-06/draft/latest    — Lấy draft Mẫu 06 mới nhất (R5.6)
  POST /api/v1/ctdt/update-cycles/design-draft    — Sinh bản nháp thiết kế cập nhật CTĐT mới (R6.0)
  POST /api/v1/ctdt/update-cycles/objectives/context-pack — Context pack mục tiêu đào tạo (R6.1A)
  POST /api/v1/ctdt/update-cycles/objectives-draft — Sinh bản nháp cập nhật mục tiêu đào tạo (R6.1B)
  POST /api/v1/ctdt/update-cycles/outcomes/context-pack — Context pack chuẩn đầu ra (R6.2A)
  POST /api/v1/ctdt/update-cycles/outcomes-draft — Sinh bản nháp cập nhật chuẩn đầu ra (R6.2B)
  GET  /api/v1/ctdt/update-cycles/{id}/alignment/objectives-outcomes — Review liên kết mục tiêu ↔ CĐR (R6.2C)
  GET  /api/v1/ctdt/mapping-draft/schema — Schema contract nháp mapping (R6.3A)
  POST /api/v1/ctdt/update-cycles/mapping-draft/build — Build nháp mapping từ objective/outcome drafts (R6.3B)
  GET  /api/v1/ctdt/update-cycles/{id}/mapping-draft/latest — Lấy nháp mapping mới nhất (R6.3C)

─── Legacy endpoints (backward-compatible) ──────────────────────────
  POST /api/v1/ctdt/query              — Truy vấn thông tin CTĐT (LLM answer)
  POST /api/v1/ctdt/review             — Rà soát nội dung + đề xuất cập nhật
  POST /api/v1/ctdt/suggest-objectives — Đề xuất mục tiêu đào tạo
  POST /api/v1/ctdt/suggest-outcomes   — Đề xuất chuẩn đầu ra
  POST /api/v1/ctdt/suggest-mapping    — Ma trận liên kết CĐR - học phần

Legacy endpoints giữ nguyên route contract và request/response schema
để đảm bảo backward compatibility với client hiện tại.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth_deps import get_current_user
from app.core.config import settings
from app.db.models.document import Document
from app.db.models.user import User
from app.db.session import get_db
from app.services import ctdt_service
from app.services.answer_service import AnswerService, AnswerSnippet
from app.services.ctdt_service import RetrievalConfidence
from app.services.ctdt_ingest_service import (
    IngestPipelineError,
    IngestValidationError,
    get_document_ctdt_info,
    ingest_from_url,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/ctdt", tags=["ctdt"])


# ──────────────────────────────────────────────────────────────────────────────
# Dependency: query service from app.state (initialised once in lifespan)
# ──────────────────────────────────────────────────────────────────────────────


def _get_query_svc(request: Request):
    return request.app.state.query_svc


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


# -----------------------------------------------------------------------------
# Legacy / Moodle-oriented endpoints
#
# Kept for backward compatibility only. Do not extend these endpoints for the
# new CTDT Processing Engine pipeline in R3.2.
# -----------------------------------------------------------------------------


@router.post("/query", response_model=CTDTQueryResponse)
async def ctdt_query(
    body: CTDTQueryRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
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
    # Giới hạn snippet đưa vào LLM bằng CTDT_LLM_MAX_SNIPPETS (không phải
    # final_limit) để tránh vượt context window và kiểm soát chi phí token.
    answer: str | None = None
    try:
        snippets = [
            AnswerSnippet(
                document_id=r.document_id,
                chunk_id=r.chunk_id,
                snippet=r.snippet,
                score=r.score,
            )
            for r in results[:settings.CTDT_LLM_MAX_SNIPPETS]
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
    query_svc=Depends(_get_query_svc),
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
        review = await ctdt_service.review_section(
            section_title=body.section_title,
            current_content=body.current_content,
            focus=focus,
            results=results,
        )
        summary = review.get("summary")
        for item in review.get("suggestions", []):
            try:
                suggestions.append(CTDTReviewSuggestion(
                    type=item.get("type", "note"),
                    description=item.get("description", ""),
                ))
            except Exception:
                continue

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


# ──────────────────────────────────────────────────────────────────────────────
# Confidence signal
# ──────────────────────────────────────────────────────────────────────────────

_CONFIDENCE_WARNINGS: dict[str, str | None] = {
    "no_reference": (
        "⚠️ Chưa có tài liệu tham chiếu trong hệ thống. "
        "Output dưới đây do AI sinh từ kiến thức chung, cần giảng viên kiểm chứng kỹ."
    ),
    "low": (
        "⚠️ Chỉ tìm được ít tài liệu liên quan (điểm thấp). "
        "Output có thể không sát với chuẩn của trường."
    ),
    "medium": "Có một số tài liệu tham chiếu. Vẫn nên kiểm tra lại với tài liệu gốc.",
    "high": None,
}


class ConfidenceInfo(BaseModel):
    level: Literal["high", "medium", "low", "no_reference"]
    retrieved_count: int
    top_score: float | None
    reason: str
    warning: str | None = None


def _build_confidence_info(confidence: RetrievalConfidence) -> ConfidenceInfo:
    """Map service-layer RetrievalConfidence to API-layer ConfidenceInfo."""
    return ConfidenceInfo(
        level=confidence.level,
        retrieved_count=confidence.retrieved_count,
        top_score=confidence.top_score,
        reason=confidence.reason,
        warning=_CONFIDENCE_WARNINGS.get(confidence.level),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Legacy / Moodle-oriented suggestion endpoints:
# suggest-objectives / suggest-outcomes / suggest-mapping
# ──────────────────────────────────────────────────────────────────────────────


class SuggestObjectivesRequest(BaseModel):
    program_name: str = Field(min_length=1, max_length=500, description="Tên ngành đào tạo")
    degree_level: str = Field(min_length=1, max_length=100, description="Trình độ đào tạo (đại học, thạc sĩ...)")
    current_content: str | None = Field(default=None, max_length=8000, description="Nội dung mục tiêu hiện tại (nếu có)")


class ObjectiveSuggestion(BaseModel):
    code: str
    type: str
    content: str
    rationale: str


class SuggestObjectivesResponse(BaseModel):
    program_name: str
    degree_level: str
    objectives: list[ObjectiveSuggestion]
    confidence: ConfidenceInfo
    latency_ms: int


class SuggestOutcomesRequest(BaseModel):
    program_name: str = Field(min_length=1, max_length=500)
    degree_level: str = Field(min_length=1, max_length=100)
    objectives: list[dict] = Field(description="Mục tiêu đã duyệt (output từ suggest-objectives)")


class OutcomeSuggestion(BaseModel):
    code: str
    category: str
    content: str
    bloom_level: str
    maps_to_objectives: list[str]


class SuggestOutcomesResponse(BaseModel):
    program_name: str
    degree_level: str
    outcomes: list[OutcomeSuggestion]
    confidence: ConfidenceInfo
    latency_ms: int


class CourseItem(BaseModel):
    code: str = Field(description="Mã học phần")
    name: str = Field(description="Tên học phần")


class SuggestMappingRequest(BaseModel):
    program_name: str = Field(min_length=1, max_length=500, description="Tên ngành đào tạo (dùng để retrieve tài liệu tham chiếu)")
    outcomes: list[dict] = Field(description="Chuẩn đầu ra (output từ suggest-outcomes)")
    courses: list[CourseItem] = Field(description="Danh sách học phần")


class MappingEntry(BaseModel):
    outcome_code: str
    course_code: str
    contribution: str


class SuggestMappingResponse(BaseModel):
    matrix: list[MappingEntry]
    coverage_gaps: list[str]
    confidence: ConfidenceInfo
    latency_ms: int


@router.post("/suggest-objectives", response_model=SuggestObjectivesResponse)
async def ctdt_suggest_objectives(
    body: SuggestObjectivesRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    Đề xuất mục tiêu đào tạo dựa trên RAG + LLM.

    Retrieve CTĐT mẫu từ knowledge base, sau đó gọi LLM với prompt
    template cứng để sinh mục tiêu có cấu trúc (code, type, content, rationale).
    """
    t0 = time.monotonic()

    objectives_raw = await ctdt_service.suggest_objectives(
        query_svc=query_svc,
        tenant_id=user.tenant_id,
        user_id=user.id,
        program_name=body.program_name,
        degree_level=body.degree_level,
        current_content=body.current_content or "",
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "ctdt.suggest_objectives tenant_id=%s program=%s count=%d retrieved_count=%d latency_ms=%d",
        user.tenant_id, body.program_name, len(objectives_raw["items"]),
        objectives_raw["retrieved_count"], elapsed_ms,
    )

    objectives = []
    for item in objectives_raw["items"]:
        try:
            objectives.append(ObjectiveSuggestion(
                code=item.get("code", ""),
                type=item.get("type", "specific"),
                content=item.get("content", ""),
                rationale=item.get("rationale", ""),
            ))
        except Exception:
            continue

    confidence_info = _build_confidence_info(objectives_raw["confidence"])

    return SuggestObjectivesResponse(
        program_name=body.program_name,
        degree_level=body.degree_level,
        objectives=objectives,
        confidence=confidence_info,
        latency_ms=elapsed_ms,
    )


@router.post("/suggest-outcomes", response_model=SuggestOutcomesResponse)
async def ctdt_suggest_outcomes(
    body: SuggestOutcomesRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    Đề xuất chuẩn đầu ra dựa trên mục tiêu đã duyệt + RAG + LLM.

    Retrieve CĐR mẫu từ knowledge base, sau đó gọi LLM để sinh CĐR
    có cấu trúc (code, category, content, bloom_level, maps_to_objectives).
    """
    t0 = time.monotonic()

    outcomes_raw = await ctdt_service.suggest_outcomes(
        query_svc=query_svc,
        tenant_id=user.tenant_id,
        user_id=user.id,
        program_name=body.program_name,
        degree_level=body.degree_level,
        objectives=body.objectives,
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "ctdt.suggest_outcomes tenant_id=%s program=%s count=%d retrieved_count=%d latency_ms=%d",
        user.tenant_id, body.program_name, len(outcomes_raw["items"]),
        outcomes_raw["retrieved_count"], elapsed_ms,
    )

    outcomes = []
    for item in outcomes_raw["items"]:
        try:
            outcomes.append(OutcomeSuggestion(
                code=item.get("code", ""),
                category=item.get("category", "knowledge"),
                content=item.get("content", ""),
                bloom_level=item.get("bloom_level", "apply"),
                maps_to_objectives=item.get("maps_to_objectives", []),
            ))
        except Exception:
            continue

    confidence_info = _build_confidence_info(outcomes_raw["confidence"])

    return SuggestOutcomesResponse(
        program_name=body.program_name,
        degree_level=body.degree_level,
        outcomes=outcomes,
        confidence=confidence_info,
        latency_ms=elapsed_ms,
    )


@router.post("/suggest-mapping", response_model=SuggestMappingResponse)
async def ctdt_suggest_mapping(
    body: SuggestMappingRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    Tạo ma trận liên kết CĐR - học phần bằng LLM.

    Input: danh sách CĐR + danh sách học phần.
    Output: mapping matrix JSON với mức đóng góp (high/medium/low)
    và danh sách CĐR chưa được hỗ trợ (coverage_gaps).
    """
    t0 = time.monotonic()

    courses_dicts = [c.model_dump() for c in body.courses]
    result = await ctdt_service.suggest_mapping(
        query_svc=query_svc,
        tenant_id=user.tenant_id,
        user_id=user.id,
        program_name=body.program_name,
        outcomes=body.outcomes,
        courses=courses_dicts,
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "ctdt.suggest_mapping tenant_id=%s program=%s outcomes=%d courses=%d matrix_entries=%d latency_ms=%d",
        user.tenant_id, body.program_name, len(body.outcomes), len(body.courses),
        len(result.get("matrix", [])), elapsed_ms,
    )

    matrix = []
    for item in result.get("matrix", []):
        try:
            matrix.append(MappingEntry(
                outcome_code=item.get("outcome_code", ""),
                course_code=item.get("course_code", ""),
                contribution=item.get("contribution", "medium"),
            ))
        except Exception:
            continue

    confidence_info = _build_confidence_info(result["confidence"])

    return SuggestMappingResponse(
        matrix=matrix,
        coverage_gaps=result.get("coverage_gaps", []),
        confidence=confidence_info,
        latency_ms=elapsed_ms,
    )


# ──────────────────────────────────────────────────────────────────────────────
# New CTDT Engine endpoints: FileServer ingest + status
# ──────────────────────────────────────────────────────────────────────────────


class DocumentRole(str, Enum):
    """Vai trò tài liệu trong quy trình cập nhật CTĐT."""
    LEGAL_REGULATION = "legal_regulation"
    DIRECTION_DECISION = "direction_decision"
    CURRENT_CURRICULUM = "current_curriculum"
    COURSE_SYLLABUS = "course_syllabus"
    SURVEY_EVIDENCE = "survey_evidence"
    MEETING_REPORT = "meeting_report"
    COMPARISON_REPORT = "comparison_report"
    TEMPLATE = "template"
    OTHER = "other"


class CTDTIngestRequest(BaseModel):
    """
    Yêu cầu ingest tài liệu CTĐT từ FileServer URL.

    CTĐT upload file → FileServer lưu file → RAG nhận URL + metadata
    → RAG tải file, extract, chunk, index → trả ai_document_id.
    """
    external_file_id: str = Field(
        ..., min_length=1, max_length=512,
        description="ID file trên FileServer (do CTĐT cung cấp)",
    )
    file_url: str = Field(
        ..., min_length=1, max_length=2048,
        description="URL tải file từ FileServer (temporary URL)",
    )
    filename: str = Field(
        ..., min_length=1, max_length=512,
        description="Tên file gốc",
    )
    mime_type: str = Field(
        ..., min_length=1, max_length=256,
        description="MIME type của file",
    )
    checksum: str | None = Field(
        None, max_length=128,
        description="Checksum file (SHA-256, tùy chọn)",
    )
    update_cycle_id: str = Field(
        ..., min_length=1, max_length=64,
        description="ID đợt cập nhật CTĐT",
    )
    program_id: str | None = Field(
        None, max_length=64,
        description="ID chương trình đào tạo",
    )
    program_code: str | None = Field(
        None, max_length=64,
        description="Mã ngành",
    )
    program_name: str | None = Field(
        None, max_length=512,
        description="Tên ngành/chương trình",
    )
    document_role: DocumentRole = Field(
        ...,
        description="Vai trò tài liệu trong quy trình cập nhật CTĐT",
    )
    uploaded_by: str | None = Field(
        None, max_length=64,
        description="Mã nhân sự người tải lên",
    )


class CTDTIngestResponse(BaseModel):
    """Response sau khi ingest tài liệu CTĐT thành công."""
    ai_document_id: int
    external_file_id: str
    ingest_status: str
    text_length: int
    chunk_count: int
    message: str


class CTDTDocumentStatusResponse(BaseModel):
    """Response trạng thái chi tiết tài liệu CTĐT."""
    ai_document_id: int
    external_file_id: str | None = None
    filename: str | None = None
    document_role: str | None = None
    update_cycle_id: str | None = None
    program_id: str | None = None
    program_code: str | None = None
    program_name: str | None = None
    ingest_status: str
    text_length: int
    chunk_count: int
    embedding_count: int | None = None
    indexed_count: int | None = None
    indexed: bool | None = None
    vector_index: str | None = None
    embedding_provider: str | None = None
    error_message: str | None = None
    uploaded_by: str | None = None
    source_system: str | None = None
    source_module: str | None = None
    mime_type: str | None = None
    checksum: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CTDTIngestErrorDetail(BaseModel):
    """Structured error detail for ingest failures."""
    code: str
    message: str
    retryable: bool


@router.post(
    "/documents/ingest",
    response_model=CTDTIngestResponse,
    responses={
        400: {"description": "Validation error (invalid_document_role, invalid_file_url, unsupported_mime_type)"},
        422: {"description": "Extract failed"},
        502: {"description": "Download failed from FileServer"},
    },
)
async def ctdt_ingest_document(
    body: CTDTIngestRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Ingest tài liệu CTĐT từ FileServer URL.

    CTĐT upload file → FileServer lưu → RAG nhận URL + metadata nghiệp vụ
    → RAG tải file, extract text, chunk, embed/index
    → Trả ai_document_id và trạng thái xử lý.
    """
    t0 = time.monotonic()

    try:
        result = await ingest_from_url(
            db,
            tenant_id=user.tenant_id,
            external_file_id=body.external_file_id,
            file_url=body.file_url,
            filename=body.filename,
            mime_type=body.mime_type,
            checksum=body.checksum,
            update_cycle_id=body.update_cycle_id,
            program_id=body.program_id,
            program_code=body.program_code,
            program_name=body.program_name,
            document_role=body.document_role.value,
            uploaded_by=body.uploaded_by,
        )
    except IngestValidationError as exc:
        logger.warning(
            "ctdt.ingest.validation_error code=%s external_file_id=%s",
            exc.error.code, body.external_file_id,
        )
        raise HTTPException(
            status_code=400,
            detail=exc.error.to_dict(),
        )
    except IngestPipelineError as exc:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.warning(
            "ctdt.ingest.pipeline_error code=%s external_file_id=%s elapsed_ms=%d",
            exc.error.code, body.external_file_id, elapsed_ms,
        )
        # Map error codes to HTTP status
        status_map = {
            "download_failed": 502,
            "file_too_large": 413,
            "unsupported_mime_type": 415,
            "extract_failed": 422,
            "index_failed": 500,
            "checksum_mismatch": 400,
        }
        http_status = status_map.get(exc.error.code, 500)
        raise HTTPException(
            status_code=http_status,
            detail=exc.error.to_dict(),
        )
    except Exception:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.exception(
            "ctdt.ingest.unexpected_error external_file_id=%s elapsed_ms=%d",
            body.external_file_id, elapsed_ms,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "index_failed",
                "message": "Lỗi nội bộ khi xử lý tài liệu",
                "retryable": True,
            },
        )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "ctdt.ingest.ok ai_document_id=%d external_file_id=%s "
        "status=%s text_length=%d chunk_count=%d elapsed_ms=%d",
        result.ai_document_id, result.external_file_id,
        result.ingest_status, result.text_length, result.chunk_count,
        elapsed_ms,
    )

    return CTDTIngestResponse(
        ai_document_id=result.ai_document_id,
        external_file_id=result.external_file_id,
        ingest_status=result.ingest_status,
        text_length=result.text_length,
        chunk_count=result.chunk_count,
        message=result.message,
    )


@router.get(
    "/documents/{ai_document_id}",
    response_model=CTDTDocumentStatusResponse,
    responses={
        404: {"description": "Document not found"},
    },
)
async def ctdt_get_document_status(
    ai_document_id: int,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Xem trạng thái tài liệu CTĐT đã ingest.

    Trả về metadata nghiệp vụ, trạng thái xử lý, số chunk, độ dài text.
    Chỉ trả document thuộc tenant của user hiện tại.
    """
    result = await db.execute(
        select(Document).where(
            Document.id == ai_document_id,
            Document.tenant_id == user.tenant_id,
        )
    )
    doc = result.scalar_one_or_none()

    if not doc:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "document_not_found",
                "message": f"Document {ai_document_id} không tìm thấy",
                "retryable": False,
            },
        )

    info = get_document_ctdt_info(doc)
    return CTDTDocumentStatusResponse(**info)


# ──────────────────────────────────────────────────────────────────────────────
# New CTDT Engine endpoint: metadata-scoped retrieval
# ──────────────────────────────────────────────────────────────────────────────


class CTDTRetrieveRequest(BaseModel):
    """
    Truy xuất tài liệu CTĐT theo phạm vi đợt cập nhật + vai trò tài liệu.

    - query: câu hỏi/truy vấn tìm kiếm
    - update_cycle_id: bắt buộc — chỉ lấy tài liệu trong đợt này
    - program_code: mã ngành (nếu có, thu hẹp phạm vi)
    - program_id: ID chương trình (nếu có)
    - task_type: loại task → quyết định policy document_roles mặc định
    - document_roles: explicit override roles (ưu tiên hơn policy)
    - top_k: số contexts trả về tối đa
    """
    query: str = Field(min_length=1, max_length=8000, description="Câu hỏi truy vấn")
    update_cycle_id: str = Field(min_length=1, max_length=64, description="ID đợt cập nhật")
    program_id: str | None = Field(default=None, max_length=64, description="ID chương trình đào tạo")
    program_code: str | None = Field(default=None, max_length=64, description="Mã ngành")
    task_type: str = Field(
        default="general_query",
        description="Loại task: general_query, evidence_analysis, current_curriculum_review, "
                    "change_proposal, objective_suggestion, outcome_suggestion, "
                    "course_structure, matrix_mapping, template_lookup, "
                    "curriculum_update_design",
    )
    document_roles: list[str] | None = Field(
        default=None,
        description="Override vai trò tài liệu: current_curriculum, survey_evidence, "
                    "meeting_report, legal_regulation, direction_decision, "
                    "comparison_report, course_syllabus, template, other",
    )
    top_k: int = Field(default=8, ge=1, le=50, description="Số contexts trả về")


class CTDTRetrieveSourceMeta(BaseModel):
    update_cycle_id: str | None = None
    program_code: str | None = None
    program_id: str | None = None
    section: str | None = None
    page: str | None = None


class CTDTRetrieveContextItem(BaseModel):
    ai_document_id: int
    external_file_id: str | None = None
    filename: str | None = None
    document_role: str | None = None
    chunk_id: int
    chunk_index: int
    score: float
    text: str
    source: CTDTRetrieveSourceMeta


class CTDTRetrieveResponse(BaseModel):
    query: str
    update_cycle_id: str
    program_code: str | None = None
    task_type: str
    document_roles_used: list[str]
    contexts: list[CTDTRetrieveContextItem]
    scoped_document_count: int
    latency_ms: int


@router.post("/retrieve", response_model=CTDTRetrieveResponse)
async def ctdt_retrieve(
    body: CTDTRetrieveRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    R3 — Truy xuất tài liệu CTĐT theo phạm vi nghiệp vụ.

    Scoped retrieval:
      - Chỉ trả tài liệu trong update_cycle_id yêu cầu.
      - Lọc theo document_role (từ policy hoặc explicit).
      - Không fallback sang đợt cập nhật khác.
      - Trả contexts = [] nếu không có tài liệu phù hợp.
    """
    from app.services.ctdt_retrieval_service import (
        CTDTTaskType,
        ctdt_retrieve as do_retrieve,
    )

    tenant_id = user.tenant_id
    user_id = user.id

    # Validate task_type
    try:
        task_type = CTDTTaskType(body.task_type)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_task_type",
                "message": (
                    f"task_type '{body.task_type}' không hợp lệ. "
                    f"Hỗ trợ: {[t.value for t in CTDTTaskType]}"
                ),
                "retryable": False,
            },
        )

    try:
        result = await do_retrieve(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            query=body.query,
            update_cycle_id=body.update_cycle_id,
            program_code=body.program_code,
            program_id=body.program_id,
            task_type=task_type,
            document_roles=body.document_roles,
            top_k=body.top_k,
            query_svc=query_svc,
        )
    except Exception as exc:
        logger.error(
            "ctdt.retrieve_failed tenant_id=%s update_cycle=%s: %s",
            tenant_id, body.update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "retrieval_error",
                "message": "Lỗi truy xuất tài liệu CTĐT.",
                "retryable": True,
            },
        )

    return CTDTRetrieveResponse(
        query=result.query,
        update_cycle_id=result.update_cycle_id,
        program_code=result.program_code,
        task_type=result.task_type,
        document_roles_used=result.document_roles_used,
        contexts=[
            CTDTRetrieveContextItem(
                ai_document_id=ctx.ai_document_id,
                external_file_id=ctx.external_file_id,
                filename=ctx.filename,
                document_role=ctx.document_role,
                chunk_id=ctx.chunk_id,
                chunk_index=ctx.chunk_index,
                score=ctx.score,
                text=ctx.text,
                source=CTDTRetrieveSourceMeta(**ctx.source),
            )
            for ctx in result.contexts
        ],
        scoped_document_count=result.scoped_document_count,
        latency_ms=result.latency_ms,
    )


# ──────────────────────────────────────────────────────────────────────────────
# R4/R5: Analyze Update Cycle (skeleton or supported draft skills)
# ──────────────────────────────────────────────────────────────────────────────


class AnalyzeUpdateCycleRequest(BaseModel):
    """
    Yêu cầu phân tích đợt cập nhật chương trình đào tạo.

    - update_cycle_id: bắt buộc — ID đợt cập nhật
    - program_code: mã ngành
    - ai_document_ids: giới hạn phạm vi phân tích (nếu cung cấp thì validate scope)
    - analysis_mode: "skeleton" hoặc "draft"; draft chạy evidence_summary,
      evaluation_points, change_proposals
    - save_draft: chỉ lưu khi analysis_mode="draft"
    - top_k_per_task: số contexts tối đa mỗi task
    """
    update_cycle_id: str = Field(min_length=1, max_length=64)
    program_id: str | None = Field(default=None, max_length=64)
    program_code: str | None = Field(default=None, max_length=64)
    program_name: str | None = Field(default=None, max_length=256)
    ai_document_ids: list[int] | None = Field(
        default=None,
        description="Giới hạn phân tích trong các tài liệu này. "
                    "Nếu cung cấp thì phải thuộc đúng update_cycle_id.",
    )
    document_roles: list[str] | None = Field(
        default=None,
        description="Override vai trò tài liệu cho tất cả tasks.",
    )
    analysis_mode: str = Field(
        default="skeleton",
        description=(
            "Chế độ phân tích: 'skeleton' (không LLM) hoặc 'draft' "
            "(gọi LLM cho evidence_summary, evaluation_points, change_proposals)."
        ),
    )
    top_k_per_task: int = Field(
        default=6, ge=1, le=50,
        description="Số contexts trả về tối đa mỗi task.",
    )
    save_draft: bool = Field(
        default=False,
        description="Nếu true và analysis_mode='draft', lưu kết quả phân tích vào draft store.",
    )


class AnalyzeSourceItem(BaseModel):
    """Một source reference từ retrieval."""
    ai_document_id: int
    external_file_id: str | None = None
    filename: str | None = None
    document_role: str | None = None
    chunk_id: int
    chunk_index: int
    score: float
    quote: str
    update_cycle_id: str | None = None
    program_code: str | None = None


class AnalyzeSkeletonItem(BaseModel):
    """Analysis item; draft mode may include generated payload."""
    status: str
    task_type: str
    sources: list[AnalyzeSourceItem]
    payload: dict[str, Any] | None = None


class AnalyzeSourceSummary(BaseModel):
    """Summary of sources used across all tasks."""
    contexts_count: int
    documents_used: list[int]
    tasks_executed: list[str]
    latency_ms: int


class AnalyzeUpdateCycleResponse(BaseModel):
    """Response from analyze-update-cycle."""
    update_cycle_id: str
    program_code: str | None = None
    program_name: str | None = None
    analysis_mode: str
    result_payload: dict[str, list[AnalyzeSkeletonItem]]
    source_summary: AnalyzeSourceSummary
    draft_id: int | None = None
    draft_saved: bool = False


class AnalysisDraftResponse(BaseModel):
    """Persisted CTDT analysis draft response."""
    draft_id: int
    update_cycle_id: str
    program_code: str | None = None
    program_name: str | None = None
    analysis_mode: str
    result_payload: dict[str, list[AnalyzeSkeletonItem]]
    source_summary: AnalyzeSourceSummary
    status: str
    created_at: datetime
    updated_at: datetime


@router.post("/update-cycles/analyze", response_model=AnalyzeUpdateCycleResponse)
async def analyze_update_cycle(
    body: AnalyzeUpdateCycleRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    R4/R5 — Phân tích đợt cập nhật CTĐT.

    Orchestrate 7 retrieval tasks, collect source contexts,
    return skeleton payload, or draft payload for the R5 supported skills.

    Nếu ai_document_ids được cung cấp, validate scope trước khi chạy.
    Nếu không có tài liệu → result_payload đủ keys nhưng list rỗng.
    """
    from app.services.ctdt_analysis_service import (
        AnalysisValidationError,
        analyze_update_cycle as do_analyze,
    )

    tenant_id = user.tenant_id
    user_id = user.id

    if body.analysis_mode not in ("skeleton", "draft"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_analysis_mode",
                "message": (
                    f"analysis_mode '{body.analysis_mode}' chưa được hỗ trợ. "
                    f"Hỗ trợ: skeleton, draft"
                ),
                "retryable": False,
            },
        )

    if body.save_draft and body.analysis_mode != "draft":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "draft_save_requires_draft_mode",
                "message": "save_draft=true chỉ hỗ trợ khi analysis_mode='draft'.",
                "retryable": False,
            },
        )

    saved_draft = None
    save_started = False

    try:
        result = await do_analyze(
            db,
            tenant_id=tenant_id,
            user_id=user_id,
            update_cycle_id=body.update_cycle_id,
            program_id=body.program_id,
            program_code=body.program_code,
            program_name=body.program_name,
            ai_document_ids=body.ai_document_ids,
            document_roles=body.document_roles,
            analysis_mode=body.analysis_mode,
            top_k_per_task=body.top_k_per_task,
            query_svc=query_svc,
        )

        if body.save_draft:
            from app.services.ctdt_analysis_draft_service import save_analysis_draft

            save_started = True
            saved_draft = await save_analysis_draft(
                db,
                tenant_id=tenant_id,
                user_id=user_id,
                result=result,
                program_id=body.program_id,
            )
            await db.commit()

    except AnalysisValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": exc.code,
                "message": exc.message,
                "retryable": False,
            },
        )
    except Exception as exc:
        if save_started:
            try:
                await db.rollback()
            except Exception:
                logger.exception(
                    "ctdt.analyze_draft_rollback_failed tenant_id=%s update_cycle=%s",
                    tenant_id, body.update_cycle_id,
                )
        logger.error(
            "ctdt.analyze_failed tenant_id=%s update_cycle=%s: %s",
            tenant_id, body.update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "analysis_error",
                "message": "Lỗi phân tích đợt cập nhật CTĐT.",
                "retryable": True,
            },
        )

    # Convert dataclass result to Pydantic response
    payload = {}
    for key, items in result.result_payload.items():
        payload[key] = [
            AnalyzeSkeletonItem(
                status=item.status,
                task_type=item.task_type,
                sources=[
                    AnalyzeSourceItem(
                        ai_document_id=s.ai_document_id,
                        external_file_id=s.external_file_id,
                        filename=s.filename,
                        document_role=s.document_role,
                        chunk_id=s.chunk_id,
                        chunk_index=s.chunk_index,
                        score=s.score,
                        quote=s.quote,
                        update_cycle_id=s.update_cycle_id,
                        program_code=s.program_code,
                    )
                    for s in item.sources
                ],
                payload=item.payload,
            )
            for item in items
        ]

    return AnalyzeUpdateCycleResponse(
        update_cycle_id=result.update_cycle_id,
        program_code=result.program_code,
        program_name=result.program_name,
        analysis_mode=result.analysis_mode,
        result_payload=payload,
        source_summary=AnalyzeSourceSummary(
            contexts_count=result.source_summary.contexts_count,
            documents_used=result.source_summary.documents_used,
            tasks_executed=result.source_summary.tasks_executed,
            latency_ms=result.source_summary.latency_ms,
        ),
        draft_id=saved_draft.id if saved_draft is not None else None,
        draft_saved=saved_draft is not None,
    )


@router.get(
    "/update-cycles/{update_cycle_id}/analysis-drafts/latest",
    response_model=AnalysisDraftResponse,
)
async def get_latest_analysis_draft(
    update_cycle_id: str,
    request: Request,
    program_code: str | None = None,
    analysis_mode: str = "draft",
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    Return the latest saved CTDT analysis draft for the current tenant.

    Drafts are isolated from official curriculum/program version data.
    """
    if analysis_mode not in ("draft", "skeleton"):
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_analysis_mode",
                "message": (
                    f"analysis_mode '{analysis_mode}' chưa được hỗ trợ. "
                    f"Hỗ trợ: skeleton, draft"
                ),
                "retryable": False,
            },
        )

    from app.services.ctdt_analysis_draft_service import get_latest_analysis_draft as get_latest

    draft = await get_latest(
        db,
        tenant_id=user.tenant_id,
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        analysis_mode=analysis_mode,
    )
    if draft is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "analysis_draft_not_found",
                "message": "Không tìm thấy draft phân tích CTĐT trong phạm vi yêu cầu.",
                "retryable": False,
            },
        )

    return AnalysisDraftResponse(
        draft_id=draft.id,
        update_cycle_id=draft.update_cycle_id,
        program_code=draft.program_code,
        program_name=draft.program_name,
        analysis_mode=draft.analysis_mode,
        result_payload=draft.result_payload,
        source_summary=draft.source_summary,
        status=draft.status,
        created_at=draft.created_at,
        updated_at=draft.updated_at,
    )


# ──────────────────────────────────────────────────────────────────────────────
# R5.6: Mẫu 06 — Dedicated draft endpoint for Laravel CTĐT
# ──────────────────────────────────────────────────────────────────────────────


# ── Mẫu 06 field names (8 fields from ChangeProposalSkill payload) ───
_MAU06_PAYLOAD_FIELDS = (
    "target_area",
    "change_type",
    "current_issue",
    "proposed_change",
    "rationale",
    "expected_impact",
    "priority",
    "confidence",
)


class Mau06DraftSourceItem(BaseModel):
    """A single source reference inside a Mẫu 06 item."""
    ai_document_id: int
    external_file_id: str | None = None
    filename: str | None = None
    document_role: str | None = None
    chunk_id: int | None = None
    chunk_index: int | None = None
    score: float | None = None
    quote: str | None = None
    update_cycle_id: str | None = None
    program_code: str | None = None


class Mau06DraftItem(BaseModel):
    """One Mẫu 06 change-proposal item, flattened for Laravel consumption."""
    target_area: str
    change_type: str
    current_issue: str
    proposed_change: str
    rationale: str
    expected_impact: str
    priority: str
    confidence: str
    sources: list[Mau06DraftSourceItem]


class Mau06DraftLatestResponse(BaseModel):
    """Response for GET /mau-06/draft/latest — flat Mẫu 06 items."""
    draft_id: int
    update_cycle_id: str
    program_code: str | None = None
    program_name: str | None = None
    analysis_mode: str
    status: str
    items: list[Mau06DraftItem]
    source_summary: AnalyzeSourceSummary
    warnings: list[str]
    created_at: datetime
    updated_at: datetime


def _extract_mau06_items_from_draft(
    draft,
) -> tuple[list[Mau06DraftItem], list[str]]:
    """
    Transform draft.result_payload["change_proposals"] into flat Mẫu 06 items.

    Returns (items, warnings). Items missing a payload dict are skipped and
    a warning is appended.
    """
    items: list[Mau06DraftItem] = []
    warnings: list[str] = []

    proposals = (draft.result_payload or {}).get("change_proposals", [])
    if not isinstance(proposals, list):
        warnings.append("change_proposals is not a list; returning empty items.")
        return items, warnings

    for idx, raw_item in enumerate(proposals):
        if not isinstance(raw_item, dict):
            warnings.append(f"Item [{idx}] is not a dict, skipped.")
            continue

        payload = raw_item.get("payload")
        if not isinstance(payload, dict):
            warnings.append(f"Item [{idx}] missing payload, skipped.")
            continue

        # Extract the 8 Mẫu 06 fields (default to empty string if missing)
        mau06_fields = {
            f: payload.get(f, "") for f in _MAU06_PAYLOAD_FIELDS
        }

        # Map sources — skip any source missing ai_document_id (no fabrication)
        raw_sources = raw_item.get("sources", [])
        source_items: list[Mau06DraftSourceItem] = []
        for src_idx, src in enumerate(raw_sources):
            if not isinstance(src, dict):
                continue
            aid = src.get("ai_document_id")
            if aid is None:
                warnings.append(
                    f"Item [{idx}] source [{src_idx}] missing ai_document_id, skipped."
                )
                continue
            source_items.append(Mau06DraftSourceItem(
                ai_document_id=aid,
                external_file_id=src.get("external_file_id"),
                filename=src.get("filename"),
                document_role=src.get("document_role"),
                chunk_id=src.get("chunk_id"),
                chunk_index=src.get("chunk_index"),
                score=src.get("score"),
                quote=src.get("quote"),
                update_cycle_id=src.get("update_cycle_id"),
                program_code=src.get("program_code"),
            ))

        items.append(Mau06DraftItem(
            **mau06_fields,
            sources=source_items,
        ))

    return items, warnings


@router.get(
    "/update-cycles/{update_cycle_id}/mau-06/draft/latest",
    response_model=Mau06DraftLatestResponse,
)
async def get_latest_mau06_draft(
    update_cycle_id: str,
    request: Request,
    program_code: str | None = None,
    analysis_mode: str = "draft",
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    R5.6 — Lấy draft Mẫu 06 mới nhất cho Laravel CTĐT.

    Đọc từ ctdt_analysis_drafts, transform change_proposals thành
    response phẳng. Read-only, không commit/rollback.
    """
    # Mẫu 06 chỉ hỗ trợ analysis_mode="draft"
    if analysis_mode != "draft":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "mau06_requires_draft_mode",
                "message": (
                    f"Endpoint Mẫu 06 chỉ hỗ trợ analysis_mode='draft'. "
                    f"Nhận: '{analysis_mode}'"
                ),
                "retryable": False,
            },
        )

    from app.services.ctdt_analysis_draft_service import get_latest_analysis_draft as get_latest

    draft = await get_latest(
        db,
        tenant_id=user.tenant_id,
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        analysis_mode="draft",
    )
    if draft is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "mau06_draft_not_found",
                "message": "Không tìm thấy draft Mẫu 06 trong phạm vi yêu cầu.",
                "retryable": False,
            },
        )

    items, warnings = _extract_mau06_items_from_draft(draft)

    return Mau06DraftLatestResponse(
        draft_id=draft.id,
        update_cycle_id=draft.update_cycle_id,
        program_code=draft.program_code,
        program_name=draft.program_name,
        analysis_mode=draft.analysis_mode,
        status=draft.status,
        items=items,
        source_summary=draft.source_summary,
        warnings=warnings,
        created_at=draft.created_at,
        updated_at=draft.updated_at,
    )


# ──────────────────────────────────────────────────────────────────────────────
# R6.0: Curriculum Update Design Draft
# ──────────────────────────────────────────────────────────────────────────────


class CurriculumDesignDraftRequest(BaseModel):
    """
    Yêu cầu sinh bản nháp thiết kế cập nhật CTĐT mới.

    Đây KHÔNG phải Mẫu 06. Đây là bản nháp đề xuất cập nhật CTĐT mới
    gồm mục tiêu, CĐR, cấu trúc, học phần, ma trận.
    """
    update_cycle_id: str = Field(min_length=1, max_length=64)
    program_id: str | None = Field(default=None, max_length=64)
    program_code: str | None = Field(default=None, max_length=64)
    program_name: str | None = Field(default=None, max_length=256)
    save_draft: bool = Field(
        default=False,
        description="Nếu true, lưu kết quả vào draft store với draft_type='curriculum_update_design'.",
    )
    top_k: int = Field(
        default=12, ge=1, le=50,
        description="Số contexts trả về tối đa.",
    )
    user_instruction: str | None = Field(
        default=None, max_length=2000,
        description="Hướng dẫn bổ sung từ hệ thống (tùy chọn).",
    )


class DesignDraftSourceSummaryResponse(BaseModel):
    """Summary of sources used in design draft generation."""
    contexts_count: int
    documents_used: list[int]
    tasks_executed: list[str]
    latency_ms: int


class CurriculumDesignDraftResponse(BaseModel):
    """Response from POST /update-cycles/design-draft."""
    update_cycle_id: str
    program_code: str | None = None
    program_name: str | None = None
    draft_type: str
    draft_id: int | None = None
    draft_saved: bool = False
    payload: dict[str, Any]
    source_summary: DesignDraftSourceSummaryResponse


@router.post(
    "/update-cycles/design-draft",
    response_model=CurriculumDesignDraftResponse,
)
async def create_curriculum_design_draft(
    body: CurriculumDesignDraftRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    R6.0 — Sinh bản nháp thiết kế cập nhật CTĐT mới.

    Đọc yêu cầu nhà trường + CTĐT cũ + minh chứng → đề xuất cải tiến
    mục tiêu, CĐR, cấu trúc, học phần, ma trận.

    Output là JSON có cấu trúc. Không ghi vào Program/ProgramVersion.
    Nếu save_draft=true → lưu vào ctdt_analysis_drafts với
    draft_type="curriculum_update_design".
    """
    from app.services.ctdt_curriculum_update_design_service import (
        generate_curriculum_update_design_draft,
    )

    try:
        result = await generate_curriculum_update_design_draft(
            db,
            tenant_id=user.tenant_id,
            user_id=user.id,
            update_cycle_id=body.update_cycle_id,
            program_id=body.program_id,
            program_code=body.program_code,
            program_name=body.program_name,
            top_k=body.top_k,
            user_instruction=body.user_instruction,
            save_draft=body.save_draft,
            query_svc=query_svc,
        )
    except Exception as exc:
        logger.error(
            "ctdt.design_draft_failed tenant_id=%s update_cycle=%s: %s",
            user.tenant_id, body.update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "design_draft_error",
                "message": "Lỗi sinh bản nháp thiết kế cập nhật CTĐT.",
                "retryable": True,
            },
        )

    return CurriculumDesignDraftResponse(
        update_cycle_id=result.update_cycle_id,
        program_code=result.program_code,
        program_name=result.program_name,
        draft_type=result.draft_type,
        draft_id=result.draft_id,
        draft_saved=result.draft_saved,
        payload=result.payload,
        source_summary=DesignDraftSourceSummaryResponse(
            contexts_count=result.source_summary.contexts_count,
            documents_used=result.source_summary.documents_used,
            tasks_executed=result.source_summary.tasks_executed,
            latency_ms=result.source_summary.latency_ms,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# R6.1A: Objective Update Context Pack (debug/internal)
# ──────────────────────────────────────────────────────────────────────────────


class ObjectiveContextPackRequest(BaseModel):
    """Yêu cầu build context pack phục vụ cập nhật mục tiêu đào tạo."""
    update_cycle_id: str = Field(min_length=1, max_length=64)
    program_id: str | None = Field(default=None, max_length=64)
    program_code: str | None = Field(default=None, max_length=64)
    program_name: str | None = Field(default=None, max_length=256)
    top_k_per_role: int = Field(
        default=5, ge=1, le=20,
        description="Số contexts tối đa mỗi role group.",
    )


class ObjectiveContextItem(BaseModel):
    ai_document_id: int
    external_file_id: str | None = None
    filename: str | None = None
    document_role: str | None = None
    chunk_id: int
    chunk_index: int
    score: float
    text: str
    source: dict[str, Any] | None = None


class ObjectiveRoleCoverage(BaseModel):
    document_roles: list[str]
    context_count: int
    documents_used: list[int]
    status: str
    scoped_document_count: int = 0
    retrieval_status: str = "ok"


class ObjectiveContextPackSourceSummary(BaseModel):
    total_contexts: int
    documents_used: list[int]
    role_groups_retrieved: list[str]
    latency_ms: int


class ObjectiveContextPackResponse(BaseModel):
    update_cycle_id: str
    program_code: str | None = None
    program_name: str | None = None
    context_pack_type: str
    role_coverage: dict[str, ObjectiveRoleCoverage]
    current_objective_contexts: list[ObjectiveContextItem]
    direction_contexts: list[ObjectiveContextItem]
    legal_contexts: list[ObjectiveContextItem]
    evidence_contexts: list[ObjectiveContextItem]
    comparison_contexts: list[ObjectiveContextItem]
    other_contexts: list[ObjectiveContextItem]
    missing_information: list[dict[str, str]]
    source_summary: ObjectiveContextPackSourceSummary


@router.post(
    "/update-cycles/objectives/context-pack",
    response_model=ObjectiveContextPackResponse,
)
async def build_objective_context_pack(
    body: ObjectiveContextPackRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    R6.1A — Build context pack cho cập nhật mục tiêu đào tạo.

    Read-only: retrieve multi-role contexts, phân nhóm, kiểm tra coverage.
    Không gọi LLM. Không lưu DB.
    """
    from app.services.ctdt_objective_context_service import (
        build_objective_update_context_pack,
    )

    try:
        pack = await build_objective_update_context_pack(
            db,
            tenant_id=user.tenant_id,
            user_id=user.id,
            update_cycle_id=body.update_cycle_id,
            program_id=body.program_id,
            program_code=body.program_code,
            program_name=body.program_name,
            top_k_per_role=body.top_k_per_role,
            query_svc=query_svc,
        )
    except Exception as exc:
        logger.error(
            "ctdt.objective_context_pack_failed tenant_id=%s update_cycle=%s: %s",
            user.tenant_id, body.update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "objective_context_pack_error",
                "message": "Lỗi build context pack mục tiêu đào tạo.",
                "retryable": True,
            },
        )

    def _map_items(items):
        return [
            ObjectiveContextItem(
                ai_document_id=it.ai_document_id,
                external_file_id=it.external_file_id,
                filename=it.filename,
                document_role=it.document_role,
                chunk_id=it.chunk_id,
                chunk_index=it.chunk_index,
                score=it.score,
                text=it.text,
                source=it.source,
            )
            for it in items
        ]

    return ObjectiveContextPackResponse(
        update_cycle_id=pack.update_cycle_id,
        program_code=pack.program_code,
        program_name=pack.program_name,
        context_pack_type=pack.context_pack_type,
        role_coverage={
            k: ObjectiveRoleCoverage(
                document_roles=v.document_roles,
                context_count=v.context_count,
                documents_used=v.documents_used,
                status=v.status,
                scoped_document_count=v.scoped_document_count,
                retrieval_status=v.retrieval_status,
            )
            for k, v in pack.role_coverage.items()
        },
        current_objective_contexts=_map_items(pack.current_objective_contexts),
        direction_contexts=_map_items(pack.direction_contexts),
        legal_contexts=_map_items(pack.legal_contexts),
        evidence_contexts=_map_items(pack.evidence_contexts),
        comparison_contexts=_map_items(pack.comparison_contexts),
        other_contexts=_map_items(pack.other_contexts),
        missing_information=pack.missing_information,
        source_summary=ObjectiveContextPackSourceSummary(
            total_contexts=pack.source_summary.total_contexts,
            documents_used=pack.source_summary.documents_used,
            role_groups_retrieved=pack.source_summary.role_groups_retrieved,
            latency_ms=pack.source_summary.latency_ms,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# R6.1B: Objective Update Draft
# ──────────────────────────────────────────────────────────────────────────────


class ObjectiveDraftRequest(BaseModel):
    """Yêu cầu sinh bản nháp cập nhật mục tiêu đào tạo."""
    update_cycle_id: str = Field(min_length=1, max_length=64)
    program_id: str | None = Field(default=None, max_length=64)
    program_code: str | None = Field(default=None, max_length=64)
    program_name: str | None = Field(default=None, max_length=256)
    save_draft: bool = Field(default=False)
    top_k_per_role: int = Field(default=5, ge=1, le=20)
    user_instruction: str | None = Field(
        default=None, max_length=2000,
        description="Hướng dẫn bổ sung cho AI khi sinh mục tiêu.",
    )
    # R6.5
    debug_context: bool = Field(
        default=False,
        description="Bật debug context để kiểm tra chunks đã dùng.",
    )
    # R6.5C: Dynamic objective count
    objective_count: int = Field(
        default=6, ge=4, le=8,
        description="Số lượng mục tiêu cụ thể cần sinh (4-8, mặc định 6).",
    )
    force_refresh: bool = Field(
        default=False,
        description="Buộc sinh lại ngay cả khi đã có draft.",
    )


class ObjectiveDraftSourceSummaryResponse(BaseModel):
    contexts_count: int
    documents_used: list[int]
    tasks_executed: list[str]
    latency_ms: int


class ObjectiveDraftResponse(BaseModel):
    update_cycle_id: str
    program_code: str | None = None
    program_name: str | None = None
    draft_type: str
    draft_id: int | None = None
    draft_saved: bool
    payload: dict[str, Any]
    context_pack_summary: dict[str, Any]
    source_summary: ObjectiveDraftSourceSummaryResponse
    generation_status: str = "needs_generation"
    warnings: list[str] = Field(default_factory=list)
    # R6.5: Flat adapted fields for Laravel
    general_objective: str = ""
    specific_objectives: list[Any] = Field(default_factory=list)
    source_summary_flat: dict[str, Any] | None = None
    debug: dict[str, Any] | None = None
    # R6.5C: Structured objective fields
    objective_count: int = 6
    format_profile: str = ""
    general_objective_text: str = ""
    specific_objective_texts: list[str] = Field(default_factory=list)
    quality_level: str = "good"
    quality_messages: list[str] = Field(default_factory=list)


@router.post(
    "/update-cycles/objectives-draft",
    response_model=ObjectiveDraftResponse,
)
async def generate_objectives_draft(
    body: ObjectiveDraftRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    R6.1B + R6.5 — Sinh bản nháp cập nhật mục tiêu đào tạo.

    Dùng context pack R6.1A → ObjectiveUpdateSkill → adapter → quality check.
    Response bao gồm cả payload gốc (backward compat) lẫn format phẳng cho Laravel.
    """
    from app.services.ctdt_objective_update_service import (
        generate_objective_update_draft,
    )

    try:
        result = await generate_objective_update_draft(
            db,
            tenant_id=user.tenant_id,
            user_id=user.id,
            update_cycle_id=body.update_cycle_id,
            program_id=body.program_id,
            program_code=body.program_code,
            program_name=body.program_name,
            top_k_per_role=body.top_k_per_role,
            user_instruction=body.user_instruction,
            save_draft=body.save_draft,
            debug_context=body.debug_context,
            query_svc=query_svc,
            objective_count=body.objective_count,
        )
    except Exception as exc:
        logger.error(
            "ctdt.objectives_draft_failed tenant_id=%s update_cycle=%s: %s",
            user.tenant_id, body.update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "objective_update_error",
                "message": "Lỗi sinh bản nháp cập nhật mục tiêu đào tạo.",
                "retryable": True,
            },
        )

    return ObjectiveDraftResponse(
        update_cycle_id=result.update_cycle_id,
        program_code=result.program_code,
        program_name=result.program_name,
        draft_type=result.draft_type,
        draft_id=result.draft_id,
        draft_saved=result.draft_saved,
        payload=result.payload,
        context_pack_summary=result.context_pack_summary,
        source_summary=ObjectiveDraftSourceSummaryResponse(
            contexts_count=result.source_summary.contexts_count,
            documents_used=result.source_summary.documents_used,
            tasks_executed=result.source_summary.tasks_executed,
            latency_ms=result.source_summary.latency_ms,
        ),
        generation_status=result.generation_status,
        warnings=result.warnings or [],
        general_objective=result.general_objective,
        specific_objectives=result.specific_objectives or [],
        source_summary_flat=result.source_summary_flat,
        debug=result.debug,
        # R6.5C fields
        objective_count=result.objective_count,
        format_profile=result.format_profile,
        general_objective_text=result.general_objective_text,
        specific_objective_texts=result.specific_objective_texts or [],
        quality_level=result.quality_level,
        quality_messages=result.quality_messages or [],
    )


# ──────────────────────────────────────────────────────────────────────────────
# R6.5: Latest Objective Draft
# ──────────────────────────────────────────────────────────────────────────────

_VALID_OBJECTIVE_DRAFT_MODES = {"design"}


class ObjectiveDraftLatestResponse(BaseModel):
    """Response phẳng cho GET latest objective draft."""
    success: bool = True
    data: dict[str, Any]


@router.get(
    "/update-cycles/{update_cycle_id}/objectives-draft/latest",
    response_model=ObjectiveDraftLatestResponse,
    responses={
        404: {"description": "Không tìm thấy objective draft"},
        422: {"description": "analysis_mode không hợp lệ"},
        500: {"description": "Lỗi DB"},
    },
)
async def get_latest_objective_draft(
    update_cycle_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    program_id: str | None = None,
    program_code: str | None = None,
    analysis_mode: str = "design",
    status: str = "draft",
):
    """
    R6.5 — Lấy objective draft mới nhất, trả format phẳng.

    Read-only. Không gọi LLM. Không rebuild context pack.

    Ưu tiên đọc "_flat" block đã lưu khi save_draft=true.
    Nếu draft cũ chưa có "_flat" → fallback adapter.

    Debug context không khả dụng ở endpoint này vì không rebuild
    context pack. Muốn xem debug context → gọi POST objectives-draft
    với debug_context=true.
    """
    # Validate analysis_mode
    if analysis_mode not in _VALID_OBJECTIVE_DRAFT_MODES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_analysis_mode",
                "message": (
                    f"analysis_mode phải là một trong: "
                    f"{', '.join(_VALID_OBJECTIVE_DRAFT_MODES)}. "
                    f"Nhận: '{analysis_mode}'"
                ),
                "retryable": False,
            },
        )

    from app.services.ctdt_analysis_draft_service import get_latest_analysis_draft

    try:
        draft = await get_latest_analysis_draft(
            db,
            tenant_id=user.tenant_id,
            update_cycle_id=update_cycle_id,
            program_id=program_id,
            program_code=program_code,
            analysis_mode=analysis_mode,
            draft_type="objective_update",
            status=status,
        )
    except Exception as exc:
        logger.error(
            "ctdt.objective_draft_latest_failed tenant_id=%s cycle=%s: %s",
            user.tenant_id, update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "objective_draft_latest_error",
                "message": "Lỗi lấy objective draft mới nhất.",
                "retryable": True,
            },
        )

    if draft is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "objective_draft_not_found",
                "message": "Không tìm thấy objective draft trong phạm vi yêu cầu.",
                "retryable": False,
            },
        )

    # ── R6.5-PATCH-1: Prefer _flat if available ──────────────────
    raw_payload = draft.result_payload or {}
    flat = raw_payload.get("_flat")

    if isinstance(flat, dict):
        # Draft was saved with _flat → use pre-computed values
        general_objective = flat.get("general_objective") or ""
        specific_objectives = flat.get("specific_objectives") or []
        source_summary = flat.get("source_summary") or {}
        warnings = flat.get("warnings") or []
        # R6.5C-HARDEN-1: metadata from _flat
        objective_count = flat.get("objective_count", 6)
        format_profile = flat.get("format_profile", "")
        quality_level = flat.get("quality_level", "good")
        quality_messages = flat.get("quality_messages", [])
        general_objective_text = flat.get(
            "general_objective_text", general_objective,
        )
        specific_objective_texts = flat.get("specific_objective_texts", [])
        evidence_quality = flat.get("evidence_quality", "moderate")
    else:
        # Old draft without _flat → fallback to adapter
        from app.services.ctdt_objective_quality_service import (
            adapt_objective_payload,
            check_objective_quality,
        )

        meta = raw_payload.get("_meta", {})
        generation_status = meta.get("generation_status", "draft")

        adapted = adapt_objective_payload(
            payload=raw_payload,
            program_name=draft.program_name,
            program_code=draft.program_code,
            generation_status=generation_status,
        )

        quality_warnings = check_objective_quality(
            general_objective=adapted.general_objective,
            specific_objectives=adapted.specific_objectives,
            program_name=draft.program_name,
        )
        general_objective = adapted.general_objective
        specific_objectives = adapted.specific_objectives
        source_summary = adapted.source_summary
        warnings = adapted.warnings + quality_warnings

        # R6.5C-HARDEN-1: safe defaults for old drafts
        objective_count = meta.get("objective_count", 6)
        format_profile = meta.get("format_profile", "")
        quality_level = "warning" if warnings else "good"
        quality_messages = list(warnings) if warnings else []
        general_objective_text = general_objective
        evidence_quality = "moderate"
        # Build specific_objective_texts from specific_objectives
        specific_objective_texts = []
        for so in specific_objectives:
            if isinstance(so, dict):
                specific_objective_texts.append(
                    f"{so.get('code', '')}. {so.get('text', '')}"
                )
            elif isinstance(so, str):
                specific_objective_texts.append(so)

    data: dict[str, Any] = {
        "draft_id": draft.id,
        "status": draft.status,
        "general_objective": general_objective,
        "specific_objectives": specific_objectives,
        "source_summary": source_summary,
        "warnings": warnings,
        "debug": None,
        "raw_payload": raw_payload,
        "created_at": str(draft.created_at) if draft.created_at else None,
        "updated_at": str(draft.updated_at) if draft.updated_at else None,
        # R6.5C-HARDEN-1: metadata fields
        "objective_count": objective_count,
        "format_profile": format_profile,
        "quality_level": quality_level,
        "quality_messages": quality_messages,
        "general_objective_text": general_objective_text,
        "specific_objective_texts": specific_objective_texts,
        "evidence_quality": evidence_quality,
    }

    return ObjectiveDraftLatestResponse(success=True, data=data)


# ──────────────────────────────────────────────────────────────────────────────
# R6.2A: Outcome Update Context Pack
# ──────────────────────────────────────────────────────────────────────────────


class OutcomeContextPackRequest(BaseModel):
    """Yêu cầu build context pack cho cập nhật chuẩn đầu ra."""
    update_cycle_id: str = Field(min_length=1, max_length=64)
    program_id: str | None = Field(default=None, max_length=64)
    program_code: str | None = Field(default=None, max_length=64)
    program_name: str | None = Field(default=None, max_length=256)
    top_k_per_role: int = Field(default=5, ge=1, le=20)


class OutcomeContextItem(BaseModel):
    ai_document_id: int
    external_file_id: str | None = None
    filename: str | None = None
    document_role: str | None = None
    chunk_id: int
    chunk_index: int
    score: float
    text: str
    source: dict[str, Any] | None = None


class OutcomeRoleCoverage(BaseModel):
    document_roles: list[str]
    context_count: int
    documents_used: list[int]
    status: str
    scoped_document_count: int = 0
    retrieval_status: str = "ok"


class OutcomeContextPackSourceSummary(BaseModel):
    total_contexts: int
    documents_used: list[int]
    role_groups_retrieved: list[str]
    latency_ms: int


class OutcomeContextPackResponse(BaseModel):
    update_cycle_id: str
    program_code: str | None = None
    program_name: str | None = None
    context_pack_type: str
    role_coverage: dict[str, OutcomeRoleCoverage]
    objective_update_contexts: list[OutcomeContextItem]
    objective_update_payload: dict[str, Any] | None = None
    current_outcome_contexts: list[OutcomeContextItem]
    current_curriculum_contexts: list[OutcomeContextItem]
    direction_contexts: list[OutcomeContextItem]
    legal_contexts: list[OutcomeContextItem]
    evidence_contexts: list[OutcomeContextItem]
    comparison_contexts: list[OutcomeContextItem]
    course_syllabus_contexts: list[OutcomeContextItem]
    other_contexts: list[OutcomeContextItem]
    missing_information: list[dict[str, str]]
    source_summary: OutcomeContextPackSourceSummary


@router.post(
    "/update-cycles/outcomes/context-pack",
    response_model=OutcomeContextPackResponse,
)
async def build_outcome_context_pack(
    body: OutcomeContextPackRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    R6.2A — Build context pack cho cập nhật chuẩn đầu ra.

    Read-only: retrieve multi-role contexts, đọc latest objective draft,
    phân nhóm, kiểm tra coverage.
    Không gọi LLM. Không lưu DB.
    """
    from app.services.ctdt_outcome_context_service import (
        build_outcome_update_context_pack,
    )

    try:
        pack = await build_outcome_update_context_pack(
            db,
            tenant_id=user.tenant_id,
            user_id=user.id,
            update_cycle_id=body.update_cycle_id,
            program_id=body.program_id,
            program_code=body.program_code,
            program_name=body.program_name,
            top_k_per_role=body.top_k_per_role,
            query_svc=query_svc,
        )
    except Exception as exc:
        logger.error(
            "ctdt.outcome_context_pack_failed tenant_id=%s update_cycle=%s: %s",
            user.tenant_id, body.update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "outcome_context_pack_error",
                "message": "Lỗi build context pack chuẩn đầu ra.",
                "retryable": True,
            },
        )

    def _map_items(items):
        return [
            OutcomeContextItem(
                ai_document_id=it.ai_document_id,
                external_file_id=it.external_file_id,
                filename=it.filename,
                document_role=it.document_role,
                chunk_id=it.chunk_id,
                chunk_index=it.chunk_index,
                score=it.score,
                text=it.text,
                source=it.source,
            )
            for it in items
        ]

    return OutcomeContextPackResponse(
        update_cycle_id=pack.update_cycle_id,
        program_code=pack.program_code,
        program_name=pack.program_name,
        context_pack_type=pack.context_pack_type,
        role_coverage={
            k: OutcomeRoleCoverage(
                document_roles=v.document_roles,
                context_count=v.context_count,
                documents_used=v.documents_used,
                status=v.status,
                scoped_document_count=v.scoped_document_count,
                retrieval_status=v.retrieval_status,
            )
            for k, v in pack.role_coverage.items()
        },
        objective_update_contexts=_map_items(pack.objective_update_contexts),
        objective_update_payload=pack.objective_update_payload,
        current_outcome_contexts=_map_items(pack.current_outcome_contexts),
        current_curriculum_contexts=_map_items(pack.current_curriculum_contexts),
        direction_contexts=_map_items(pack.direction_contexts),
        legal_contexts=_map_items(pack.legal_contexts),
        evidence_contexts=_map_items(pack.evidence_contexts),
        comparison_contexts=_map_items(pack.comparison_contexts),
        course_syllabus_contexts=_map_items(pack.course_syllabus_contexts),
        other_contexts=_map_items(pack.other_contexts),
        missing_information=pack.missing_information,
        source_summary=OutcomeContextPackSourceSummary(
            total_contexts=pack.source_summary.total_contexts,
            documents_used=pack.source_summary.documents_used,
            role_groups_retrieved=pack.source_summary.role_groups_retrieved,
            latency_ms=pack.source_summary.latency_ms,
        ),
    )


# ──────────────────────────────────────────────────────────────────────────────
# R6.2B: Outcome Update Draft
# ──────────────────────────────────────────────────────────────────────────────


class OutcomeDraftRequest(BaseModel):
    """Yêu cầu sinh bản nháp cập nhật chuẩn đầu ra."""
    update_cycle_id: str = Field(min_length=1, max_length=64)
    program_id: str | None = Field(default=None, max_length=64)
    program_code: str | None = Field(default=None, max_length=64)
    program_name: str | None = Field(default=None, max_length=256)
    save_draft: bool = False
    top_k_per_role: int = Field(default=5, ge=1, le=20)
    user_instruction: str | None = Field(default=None, max_length=2000)
    # R6.8A: outcome count & allocation
    outcome_count: int = Field(
        default=10, ge=6, le=15,
        description="Số lượng CĐR cần sinh (6-15, mặc định 10).",
    )
    group_allocation: dict[str, int] | None = Field(
        default=None,
        description=(
            "Phân bổ CĐR theo nhóm. Phải khớp bảng cố định CTĐT. "
            "Nếu None → dùng bảng mặc định."
        ),
    )
    # R6.8A-PATCH-1 FIX 4: approved objective snapshot
    approved_objective_snapshot: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Snapshot mục tiêu đào tạo đã lưu/hoàn thành từ Laravel. "
            "Phải có is_completed=true, general_objective, specific_objectives."
        ),
    )
    # Legacy: approved objectives list (backward compat)
    approved_objectives: list[dict] | None = Field(
        default=None,
        description="Legacy: danh sách mục tiêu đã duyệt. Ưu tiên snapshot nếu có.",
    )


class OutcomeDraftSourceSummaryResponse(BaseModel):
    contexts_count: int
    documents_used: list[int]
    tasks_executed: list[str]
    latency_ms: int


class OutcomeDraftResponse(BaseModel):
    update_cycle_id: str
    program_code: str | None = None
    program_name: str | None = None
    draft_type: str
    draft_id: int | None = None
    draft_saved: bool
    payload: dict[str, Any]
    context_pack_summary: dict[str, Any]
    source_summary: OutcomeDraftSourceSummaryResponse
    generation_status: str = "needs_generation"
    warnings: list[str] = Field(default_factory=list)
    # R6.8A-PATCH-1: quality gate + structured flat output
    outcome_count: int = 10
    group_allocation: dict[str, int] = Field(default_factory=dict)
    outcomes_flat: list[dict[str, Any]] = Field(default_factory=list)
    outcomes_structured: list[dict[str, Any]] = Field(default_factory=list)
    outcome_texts: list[str] = Field(default_factory=list)
    objective_source: str = "none"
    format_profile: str = "tay_nguyen_mau_07"
    quality_level: str = "good"
    quality_messages: list[str] = Field(default_factory=list)


@router.post(
    "/update-cycles/outcomes-draft",
    response_model=OutcomeDraftResponse,
)
async def generate_outcomes_draft(
    body: OutcomeDraftRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    query_svc=Depends(_get_query_svc),
):
    """
    R6.2B + R6.8A-PATCH-1 — Sinh bản nháp cập nhật chuẩn đầu ra.

    Dùng OutcomeUpdateContextPack (R6.2A) + LLM để đề xuất CĐR mới.
    Hỗ trợ outcome_count, approved_objective_snapshot, quality gate.
    Không ghi dữ liệu chính thức.
    """
    from app.services.ctdt_outcome_update_service import generate_outcome_update_draft
    from app.services.ctdt_skills.outcome_update_skill import (
        _compute_default_allocation, _VALID_GROUPS, _OUTCOME_ALLOCATION_TABLE,
    )

    # R6.8A-PATCH-1 FIX 9: strict group_allocation validation
    if body.group_allocation is not None:
        invalid_keys = set(body.group_allocation.keys()) - _VALID_GROUPS
        missing_keys = _VALID_GROUPS - set(body.group_allocation.keys())
        if invalid_keys or missing_keys:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_group_allocation_keys",
                    "message": (
                        f"group_allocation phải có đúng 3 keys: {sorted(_VALID_GROUPS)}."
                    ),
                    "retryable": False,
                },
            )
        if any(v < 0 for v in body.group_allocation.values()):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "group_allocation_negative_value",
                    "message": "Giá trị trong group_allocation không được âm.",
                    "retryable": False,
                },
            )
        expected = _OUTCOME_ALLOCATION_TABLE.get(body.outcome_count)
        if expected and body.group_allocation != expected:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "group_allocation_contract_mismatch",
                    "message": (
                        "Phân bổ chuẩn đầu ra không khớp cấu trúc CTĐT bắt buộc "
                        f"cho số lượng đã chọn. Kỳ vọng: {expected}"
                    ),
                    "retryable": False,
                },
            )

    try:
        result = await generate_outcome_update_draft(
            db,
            tenant_id=user.tenant_id,
            user_id=user.id,
            update_cycle_id=body.update_cycle_id,
            program_id=body.program_id,
            program_code=body.program_code,
            program_name=body.program_name,
            top_k_per_role=body.top_k_per_role,
            user_instruction=body.user_instruction,
            save_draft=body.save_draft,
            query_svc=query_svc,
            outcome_count=body.outcome_count,
            group_allocation=body.group_allocation,
            approved_objectives=body.approved_objectives,
            approved_objective_snapshot=body.approved_objective_snapshot,
        )
    except Exception as exc:
        logger.error(
            "ctdt.outcome_draft_failed tenant_id=%s update_cycle=%s: %s",
            user.tenant_id, body.update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "outcome_update_error",
                "message": "Lỗi sinh bản nháp cập nhật chuẩn đầu ra.",
                "retryable": True,
            },
        )

    return OutcomeDraftResponse(
        update_cycle_id=result.update_cycle_id,
        program_code=result.program_code,
        program_name=result.program_name,
        draft_type=result.draft_type,
        draft_id=result.draft_id,
        draft_saved=result.draft_saved,
        payload=result.payload,
        context_pack_summary=result.context_pack_summary,
        source_summary=OutcomeDraftSourceSummaryResponse(
            contexts_count=result.source_summary.contexts_count,
            documents_used=result.source_summary.documents_used,
            tasks_executed=result.source_summary.tasks_executed,
            latency_ms=result.source_summary.latency_ms,
        ),
        generation_status=result.generation_status,
        warnings=result.warnings or [],
        outcome_count=result.outcome_count,
        group_allocation=result.group_allocation or {},
        outcomes_flat=result.outcomes_flat or [],
        outcomes_structured=result.outcomes_structured or [],
        outcome_texts=result.outcome_texts or [],
        objective_source=result.objective_source,
        quality_level=result.quality_level,
        quality_messages=result.quality_messages or [],
    )


# ──────────────────────────────────────────────────────────────────────────────
# R6.2C: Objective ↔ Outcome Alignment Review
# ──────────────────────────────────────────────────────────────────────────────


class AlignmentReviewSummary(BaseModel):
    objectives_count: int = 0
    outcomes_count: int = 0
    covered_objectives_count: int = 0
    partially_covered_objectives_count: int = 0
    not_covered_objectives_count: int = 0
    unmapped_outcomes_count: int = 0
    low_confidence_outcomes_count: int = 0
    needs_human_review_count: int = 0


class ObjectiveCoverageItem(BaseModel):
    objective_code: str
    objective_content: str
    mapped_outcomes: list[str]
    coverage_status: Literal["covered", "partially_covered", "not_covered", "unknown"]
    issues: list[str]
    recommendation: str


class OutcomeMappingQualityItem(BaseModel):
    outcome_code: str
    outcome_content: str
    mapped_objectives: list[str]
    mapping_status: Literal["valid", "weak", "missing", "unknown"]
    confidence: Literal["low", "medium", "high"]
    quality_flags: list[str]
    issues: list[str]
    recommendation: str


class AlignmentGapItem(BaseModel):
    type: Literal[
        "objective_not_covered", "outcome_unmapped",
        "low_confidence", "missing_evidence", "needs_human_review",
    ]
    severity: Literal["low", "medium", "high"]
    description: str
    related_objective_code: str | None = None
    related_outcome_code: str | None = None


class AlignmentMissingInfo(BaseModel):
    type: str
    description: str


class AlignmentNextAction(BaseModel):
    action: str
    owner_hint: Literal["bo_mon", "khoa", "hoi_dong", "admin", "unknown"]
    priority: Literal["low", "medium", "high"]


class AlignmentReviewResponse(BaseModel):
    """Response for GET /alignment/objectives-outcomes (R6.2C)."""
    update_cycle_id: str
    program_code: str | None = None
    review_type: str = "objective_outcome_alignment"
    objective_draft_id: int | None = None
    outcome_draft_id: int | None = None
    summary: AlignmentReviewSummary
    objective_coverage: list[ObjectiveCoverageItem]
    outcome_mapping_quality: list[OutcomeMappingQualityItem]
    gaps: list[AlignmentGapItem]
    missing_information: list[AlignmentMissingInfo]
    next_actions: list[AlignmentNextAction]


@router.get(
    "/update-cycles/{update_cycle_id}/alignment/objectives-outcomes",
    response_model=AlignmentReviewResponse,
)
async def review_alignment_objectives_outcomes(
    update_cycle_id: str,
    request: Request,
    program_code: str | None = None,
    program_id: str | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    R6.2C — Review liên kết mục tiêu đào tạo ↔ chuẩn đầu ra.

    Read-only rule-based review:
      - Đọc latest objective_update draft và outcome_update draft.
      - Đánh giá coverage, mapping quality, gaps.
      - Không gọi LLM. Không commit DB. Không save draft.
      - Nếu thiếu draft → missing_information, không crash.
    """
    from app.services.ctdt_alignment_review_service import (
        review_objective_outcome_alignment,
    )

    try:
        result = await review_objective_outcome_alignment(
            db,
            tenant_id=user.tenant_id,
            update_cycle_id=update_cycle_id,
            program_code=program_code,
            program_id=program_id,
        )
    except Exception as exc:
        logger.error(
            "ctdt.alignment_review_failed tenant_id=%s update_cycle=%s: %s",
            user.tenant_id, update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "alignment_review_error",
                "message": "Lỗi review liên kết mục tiêu ↔ chuẩn đầu ra.",
                "retryable": True,
            },
        )

    return AlignmentReviewResponse(**result)


# ──────────────────────────────────────────────────────────────────────────────
# R6.3A: Mapping Draft Schema (debug/internal)
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/mapping-draft/schema")
async def mapping_draft_schema(
    request: Request,
    user: User = Depends(get_current_user),
):
    """
    R6.3A — Introspect mapping draft contract.

    Returns supported mapping types and contribution level labels.
    Read-only, no DB, no LLM.
    """
    from app.services.ctdt_mapping_draft_contract import (
        CONTRIBUTION_LEVEL_LABELS,
        MappingDraftType,
        MappingConfidence,
        MappingSourceType,
        MappingStatus,
    )

    return {
        "draft_type": "mapping_draft",
        "mapping_types": sorted(MappingDraftType.ALL),
        "contribution_levels": {
            str(k): v for k, v in CONTRIBUTION_LEVEL_LABELS.items()
        },
        "confidence_levels": sorted(MappingConfidence.ALL),
        "source_types": sorted(MappingSourceType.ALL),
        "statuses": sorted(MappingStatus.ALL),
    }


# ──────────────────────────────────────────────────────────────────────────────
# R6.3B: Mapping Draft Builder V1
# ──────────────────────────────────────────────────────────────────────────────


class MappingDraftBuildRequest(BaseModel):
    """Yêu cầu build mapping draft từ objective/outcome drafts."""
    update_cycle_id: str = Field(min_length=1, max_length=64)
    program_id: str | None = Field(default=None, max_length=64)
    program_code: str | None = Field(default=None, max_length=64)
    program_name: str | None = Field(default=None, max_length=256)
    mapping_types: list[str] = Field(default_factory=lambda: ["objective_outcome"])
    save_draft: bool = False


class MappingDraftBuildResponse(BaseModel):
    update_cycle_id: str
    program_id: str | None = None
    program_code: str | None = None
    program_name: str | None = None
    draft_type: str = "mapping_draft"
    draft_id: int | None = None
    draft_saved: bool = False
    payload: dict[str, Any]
    source_summary: dict[str, Any]
    missing_information: list[dict[str, str]]
    warnings: list[str] = Field(default_factory=list)


@router.post(
    "/update-cycles/mapping-draft/build",
    response_model=MappingDraftBuildResponse,
)
async def build_mapping_draft_endpoint(
    body: MappingDraftBuildRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    R6.3B — Build mapping draft từ objective/outcome drafts.

    V1: chỉ build objective_outcome mapping.
    Không gọi LLM. Không ghi mapping chính thức.
    Nếu save_draft=true → lưu vào ctdt_analysis_drafts.
    """
    from app.services.ctdt_mapping_draft_builder_service import (
        build_mapping_draft,
        MappingDraftSaveError,
    )

    try:
        result = await build_mapping_draft(
            db,
            tenant_id=user.tenant_id,
            user_id=user.id,
            update_cycle_id=body.update_cycle_id,
            program_id=body.program_id,
            program_code=body.program_code,
            program_name=body.program_name,
            save_draft=body.save_draft,
            mapping_types=body.mapping_types,
        )
    except MappingDraftSaveError as exc:
        logger.error(
            "ctdt.mapping_draft_save_failed tenant_id=%s update_cycle=%s: %s",
            user.tenant_id, body.update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "mapping_draft_save_error",
                "message": "Không lưu được mapping draft. Vui lòng thử lại.",
                "retryable": True,
            },
        )
    except Exception as exc:
        logger.error(
            "ctdt.mapping_draft_build_failed tenant_id=%s update_cycle=%s: %s",
            user.tenant_id, body.update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "mapping_draft_build_error",
                "message": "Lỗi build mapping draft.",
                "retryable": True,
            },
        )

    return MappingDraftBuildResponse(
        update_cycle_id=result.update_cycle_id,
        program_id=result.program_id,
        program_code=result.program_code,
        program_name=result.program_name,
        draft_type=result.draft_type,
        draft_id=result.draft_id,
        draft_saved=result.draft_saved,
        payload=result.payload,
        source_summary=result.source_summary,
        missing_information=result.missing_information,
        warnings=result.warnings,
    )


# ──────────────────────────────────────────────────────────────────────────────
# R6.3C: Mapping Draft Latest
# ──────────────────────────────────────────────────────────────────────────────


class MappingDraftLatestResponse(BaseModel):
    draft_id: int
    update_cycle_id: str
    program_id: str | None = None
    program_code: str | None = None
    program_name: str | None = None
    draft_type: str = "mapping_draft"
    analysis_mode: str = "design"
    status: str = "draft"
    payload: dict[str, Any]
    source_summary: dict[str, Any]
    created_at: str | None = None
    updated_at: str | None = None


@router.get(
    "/update-cycles/{update_cycle_id}/mapping-draft/latest",
    response_model=MappingDraftLatestResponse,
)
async def get_latest_mapping_draft(
    update_cycle_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
    program_id: str | None = None,
    program_code: str | None = None,
    analysis_mode: str = "design",
    status: str = "draft",
):
    """
    R6.3C — Lấy mapping draft mới nhất cho Laravel render/edit.

    Read-only. Không gọi LLM. Không ghi mapping chính thức.
    """
    if analysis_mode != "design":
        raise HTTPException(
            status_code=422,
            detail={
                "code": "mapping_draft_requires_design_mode",
                "message": "Mapping draft chỉ hỗ trợ analysis_mode='design'.",
                "retryable": False,
            },
        )

    from app.services.ctdt_analysis_draft_service import get_latest_analysis_draft

    try:
        draft = await get_latest_analysis_draft(
            db,
            tenant_id=user.tenant_id,
            update_cycle_id=update_cycle_id,
            program_id=program_id,
            program_code=program_code,
            analysis_mode=analysis_mode,
            draft_type="mapping_draft",
            status=status,
        )
    except Exception as exc:
        logger.error(
            "ctdt.mapping_draft_latest_failed tenant_id=%s cycle=%s: %s",
            user.tenant_id, update_cycle_id, exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "mapping_draft_latest_error",
                "message": "Lỗi lấy mapping draft mới nhất.",
                "retryable": True,
            },
        )

    if draft is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "mapping_draft_not_found",
                "message": "Không tìm thấy mapping draft trong phạm vi yêu cầu.",
                "retryable": False,
            },
        )

    return MappingDraftLatestResponse(
        draft_id=draft.id,
        update_cycle_id=draft.update_cycle_id,
        program_id=draft.program_id,
        program_code=draft.program_code,
        program_name=draft.program_name,
        draft_type=draft.draft_type,
        analysis_mode=draft.analysis_mode,
        status=draft.status,
        payload=draft.result_payload or {},
        source_summary=draft.source_summary or {},
        created_at=str(draft.created_at) if draft.created_at else None,
        updated_at=str(draft.updated_at) if draft.updated_at else None,
    )
