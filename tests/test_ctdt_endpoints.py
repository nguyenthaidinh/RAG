"""Tests for CTDT-specific endpoints."""
import pytest
from unittest.mock import AsyncMock, patch
from app.api.v1.ctdt import CTDTQueryRequest, MoodleCourseContext


class TestCTDTQueryRequest:
    def test_valid_request(self):
        req = CTDTQueryRequest(question="Mục tiêu đào tạo ngành CNTT?")
        assert req.question == "Mục tiêu đào tạo ngành CNTT?"
        assert req.mode == "hybrid"
        assert req.final_limit == 10

    def test_with_moodle_context(self):
        req = CTDTQueryRequest(
            question="Chuẩn đầu ra là gì?",
            context=MoodleCourseContext(
                course_id=101,
                course_fullname="Công nghệ thông tin",
            ),
        )
        assert req.context.course_fullname == "Công nghệ thông tin"

    def test_question_too_short(self):
        with pytest.raises(Exception):
            CTDTQueryRequest(question="")


class TestCTDTIntentDetection:
    """Test CTDT-specific intent detection (after C1 improvement)."""

    def test_objective_intent(self):
        from app.services.answer_service import AnswerService
        svc = AnswerService()
        assert svc._detect_intent("Mục tiêu đào tạo ngành CNTT là gì?") == "ctdt_objective"

    def test_outcome_intent(self):
        from app.services.answer_service import AnswerService
        svc = AnswerService()
        assert svc._detect_intent("Chuẩn đầu ra của chương trình?") == "ctdt_outcome"

    def test_regulation_intent(self):
        from app.services.answer_service import AnswerService
        svc = AnswerService()
        assert svc._detect_intent("Thông tư 17 quy định gì?") == "ctdt_regulation"

    def test_mapping_intent(self):
        from app.services.answer_service import AnswerService
        svc = AnswerService()
        assert svc._detect_intent("Ma trận đóng góp của học phần?") == "ctdt_mapping"

    def test_general_intent_fallback(self):
        from app.services.answer_service import AnswerService
        svc = AnswerService()
        assert svc._detect_intent("Chương trình này dạy gì?") == "general"

    def test_overview_intent_not_overridden(self):
        from app.services.answer_service import AnswerService
        svc = AnswerService()
        assert svc._detect_intent("Tóm tắt chương trình đào tạo") == "overview"


class TestRetrievalConfidence:
    """Test _compute_confidence logic."""

    def test_no_reference_when_count_zero(self):
        from app.services.ctdt_service import _compute_confidence
        c = _compute_confidence(0, None, None)
        assert c.level == "no_reference"

    def test_no_reference_when_failure(self):
        from app.services.ctdt_service import _compute_confidence
        c = _compute_confidence(3, 0.8, "connection_error")
        assert c.level == "no_reference"
        assert c.reason == "connection_error"

    def test_low_few_weak_results(self):
        from app.services.ctdt_service import _compute_confidence
        c = _compute_confidence(1, 0.2, None)
        assert c.level == "low"

    def test_low_single_good_score(self):
        """1 doc with score 0.5 → low (not enough docs for medium)."""
        from app.services.ctdt_service import _compute_confidence
        c = _compute_confidence(1, 0.5, None)
        assert c.level == "low"

    def test_medium_partial_references(self):
        from app.services.ctdt_service import _compute_confidence
        c = _compute_confidence(2, 0.4, None)
        assert c.level == "medium"

    def test_medium_two_docs_high_score(self):
        """2 docs with score 0.7 → medium (not enough docs for high)."""
        from app.services.ctdt_service import _compute_confidence
        c = _compute_confidence(2, 0.7, None)
        assert c.level == "medium"

    def test_high_sufficient_references(self):
        from app.services.ctdt_service import _compute_confidence
        c = _compute_confidence(3, 0.7, None)
        assert c.level == "high"
        assert c.top_score == 0.7

    def test_high_many_references(self):
        from app.services.ctdt_service import _compute_confidence
        c = _compute_confidence(5, 0.9, None)
        assert c.level == "high"


class TestConfidenceInfo:
    """Test ConfidenceInfo schema and warning mapping."""

    def test_no_reference_has_warning(self):
        from app.services.ctdt_service import RetrievalConfidence
        from app.api.v1.ctdt import _build_confidence_info
        c = RetrievalConfidence(level="no_reference", retrieved_count=0, top_score=None, reason="no_results")
        info = _build_confidence_info(c)
        assert info.level == "no_reference"
        assert info.warning is not None
        assert "kiểm chứng" in info.warning

    def test_low_has_warning(self):
        from app.services.ctdt_service import RetrievalConfidence
        from app.api.v1.ctdt import _build_confidence_info
        c = RetrievalConfidence(level="low", retrieved_count=1, top_score=0.2, reason="few_or_weak_references")
        info = _build_confidence_info(c)
        assert info.level == "low"
        assert info.warning is not None

    def test_high_no_warning(self):
        from app.services.ctdt_service import RetrievalConfidence
        from app.api.v1.ctdt import _build_confidence_info
        c = RetrievalConfidence(level="high", retrieved_count=3, top_score=0.7, reason="sufficient_references")
        info = _build_confidence_info(c)
        assert info.level == "high"
        assert info.warning is None
