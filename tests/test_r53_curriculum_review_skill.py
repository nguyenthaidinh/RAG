"""
Tests for R5.3 — CurrentCurriculumReviewSkill.

Covers:
1. No contexts → insufficient_evidence
2. LLM disabled → needs_generation, no fabrication
3. LLM enabled + valid JSON → generated, sources mapped
4. source_indices invalid → warning, no fake sources
5. LLM error → failed, analysis doesn't crash
6. Draft mode: evidence_summary + evaluation_points skills, 5 others skeleton
7. Skeleton mode → R4 behavior preserved
8. Full suite backward compat
"""
import json
import pytest
from unittest.mock import AsyncMock, patch

from app.services.ctdt_analysis_service import (
    ALL_PAYLOAD_KEYS,
    AnalysisSkeletonItem,
    AnalysisSource,
    analyze_update_cycle,
    _run_curriculum_review_skill,
)
from app.services.ctdt_skills.current_curriculum_review_skill import (
    CurrentCurriculumReviewResult,
    CurrentCurriculumReviewSkill,
    CurriculumEvaluationPoint,
    ReviewConfidence,
    ReviewStatus,
    _build_user_prompt,
)
from app.services.ctdt_skills.evidence_analysis_skill import (
    EvidenceAnalysisResult,
    EvidenceAnalysisSkill,
)
from app.services.ctdt_retrieval_service import (
    CTDTRetrievalContext,
    CTDTRetrievalResult,
    CTDTTaskType,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_source(
    *,
    doc_id: int = 42,
    chunk_id: int = 4200001,
    score: float = 0.88,
    quote: str = "CTĐT ngành CNTT gồm 130 tín chỉ, 45 học phần bắt buộc.",
    filename: str = "ctdt_hien_hanh.pdf",
    document_role: str = "current_curriculum",
) -> AnalysisSource:
    return AnalysisSource(
        ai_document_id=doc_id,
        external_file_id=f"file_{doc_id}",
        filename=filename,
        document_role=document_role,
        chunk_id=chunk_id,
        chunk_index=chunk_id % 100_000,
        score=score,
        quote=quote,
        update_cycle_id="15",
        program_code="7480201",
    )


def _make_retrieval_context(doc_id=42, chunk_id=4200001, score=0.88):
    return CTDTRetrievalContext(
        ai_document_id=doc_id,
        external_file_id=f"file_{doc_id}",
        filename="ctdt_hien_hanh.pdf",
        document_role="current_curriculum",
        chunk_id=chunk_id,
        chunk_index=chunk_id % 100_000,
        score=score,
        text="CTĐT ngành CNTT gồm 130 tín chỉ, 45 học phần bắt buộc.",
        source={"update_cycle_id": "15", "program_code": "7480201",
                "program_id": None, "section": None, "page": None},
    )


def _make_retrieval_result(task_type, contexts=None):
    from app.services.ctdt_analysis_service import ANALYSIS_TASK_SEEDS
    if contexts is None:
        contexts = [_make_retrieval_context()]
    return CTDTRetrievalResult(
        query=ANALYSIS_TASK_SEEDS.get(task_type, "test"),
        update_cycle_id="15",
        program_code="7480201",
        task_type=task_type.value,
        document_roles_used=[],
        contexts=contexts,
        scoped_document_count=len(set(c.ai_document_id for c in contexts)),
        latency_ms=10,
    )


# ══════════════════════════════════════════════════════════════════════
# 1. No contexts → insufficient_evidence
# ══════════════════════════════════════════════════════════════════════


class TestNoContexts:

    @pytest.mark.asyncio
    async def test_empty_sources_returns_insufficient(self):
        skill = CurrentCurriculumReviewSkill()
        result = await skill.run(update_cycle_id="15", sources=[])
        assert result.status == ReviewStatus.INSUFFICIENT_EVIDENCE.value
        assert result.items == []
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_empty_sources_no_fabrication(self):
        skill = CurrentCurriculumReviewSkill()
        result = await skill.run(update_cycle_id="15", sources=[])
        assert len(result.items) == 0


# ══════════════════════════════════════════════════════════════════════
# 2. LLM disabled → needs_generation
# ══════════════════════════════════════════════════════════════════════


class TestLLMDisabled:

    @pytest.mark.asyncio
    async def test_synthesis_disabled_returns_needs_generation(self):
        skill = CurrentCurriculumReviewSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.current_curriculum_review_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = False
            result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.status == ReviewStatus.NEEDS_GENERATION.value
        assert result.items == []
        assert any("SYNTHESIS_ENABLED=false" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_synthesis_disabled_no_fabrication(self):
        skill = CurrentCurriculumReviewSkill()
        sources = [_make_source(), _make_source(doc_id=43)]

        with patch("app.services.ctdt_skills.current_curriculum_review_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = False
            result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.items == []

    @pytest.mark.asyncio
    async def test_no_api_key_returns_needs_generation(self):
        skill = CurrentCurriculumReviewSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.current_curriculum_review_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = ""
            result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.status == ReviewStatus.NEEDS_GENERATION.value
        assert any("OPENAI_API_KEY" in w for w in result.warnings)


# ══════════════════════════════════════════════════════════════════════
# 3. LLM enabled + valid JSON → generated
# ══════════════════════════════════════════════════════════════════════


class TestLLMEnabledValidJSON:

    def test_parse_valid_response(self):
        skill = CurrentCurriculumReviewSkill()
        sources = [_make_source(), _make_source(doc_id=43)]

        raw_json = json.dumps({
            "items": [
                {
                    "aspect": "Mục tiêu đào tạo",
                    "finding": "Mục tiêu đào tạo phù hợp với nhu cầu xã hội",
                    "recommendation": "Cân nhắc bổ sung mục tiêu về AI/ML",
                    "rationale": "Xu hướng công nghệ thay đổi nhanh",
                    "confidence": "high",
                    "source_indices": [0],
                },
                {
                    "aspect": "Khối lượng tín chỉ",
                    "finding": "130 tín chỉ, đạt chuẩn TT17",
                    "recommendation": "Xem xét giảm tín chỉ tự chọn để tăng chuyên ngành",
                    "rationale": "So sánh với CTĐT tham khảo",
                    "confidence": "medium",
                    "source_indices": [0, 1],
                },
            ],
            "warnings": [],
        })

        result = skill._parse_response(raw_json, sources)
        assert result.status == "generated"
        assert len(result.items) == 2
        assert result.items[0].aspect == "Mục tiêu đào tạo"
        assert result.items[0].confidence == "high"
        assert len(result.items[0].sources) == 1
        assert result.items[1].aspect == "Khối lượng tín chỉ"
        assert len(result.items[1].sources) == 2

    def test_parse_sources_mapped_correctly(self):
        skill = CurrentCurriculumReviewSkill()
        s1 = _make_source(doc_id=42)
        s2 = _make_source(doc_id=43)

        raw_json = json.dumps({
            "items": [{
                "aspect": "CĐR",
                "finding": "test",
                "recommendation": "test",
                "rationale": "test",
                "confidence": "high",
                "source_indices": [1],
            }],
            "warnings": [],
        })

        result = skill._parse_response(raw_json, [s1, s2])
        assert len(result.items[0].sources) == 1
        assert result.items[0].sources[0].ai_document_id == 43


# ══════════════════════════════════════════════════════════════════════
# 4. source_indices invalid
# ══════════════════════════════════════════════════════════════════════


class TestInvalidSourceIndices:

    def test_out_of_range_adds_warning(self):
        skill = CurrentCurriculumReviewSkill()
        sources = [_make_source()]

        raw_json = json.dumps({
            "items": [{
                "aspect": "Test",
                "finding": "test",
                "recommendation": "test",
                "rationale": "test",
                "confidence": "medium",
                "source_indices": [0, 999],
            }],
            "warnings": [],
        })

        result = skill._parse_response(raw_json, sources)
        assert len(result.items) == 1
        assert len(result.items[0].sources) == 1  # only index 0 valid
        assert any("999" in w for w in result.warnings)

    def test_no_fake_sources_created(self):
        skill = CurrentCurriculumReviewSkill()
        sources = [_make_source()]

        raw_json = json.dumps({
            "items": [{
                "aspect": "Test",
                "finding": "test",
                "recommendation": "test",
                "rationale": "test",
                "confidence": "low",
                "source_indices": [5, 10],  # all invalid
            }],
            "warnings": [],
        })

        result = skill._parse_response(raw_json, sources)
        assert result.items[0].sources == []  # no fake sources

    def test_invalid_confidence_defaults_to_medium(self):
        skill = CurrentCurriculumReviewSkill()
        raw_json = json.dumps({
            "items": [{
                "aspect": "Test",
                "finding": "test",
                "recommendation": "test",
                "rationale": "test",
                "confidence": "super_high",
                "source_indices": [],
            }],
            "warnings": [],
        })
        result = skill._parse_response(raw_json, [])
        assert result.items[0].confidence == "medium"


# ══════════════════════════════════════════════════════════════════════
# 5. LLM error → failed
# ══════════════════════════════════════════════════════════════════════


class TestLLMError:

    @pytest.mark.asyncio
    async def test_llm_call_failure_returns_failed(self):
        skill = CurrentCurriculumReviewSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.current_curriculum_review_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "test-key"
            ms.SYNTHESIS_MODEL = "gpt-4o-mini"
            ms.SYNTHESIS_TIMEOUT_S = 5.0
            ms.SYNTHESIS_MAX_TOKENS = 1000
            ms.SYNTHESIS_TEMPERATURE = 0.15

            with patch.object(skill, "_call_openai", side_effect=RuntimeError("API error")):
                result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.status == ReviewStatus.FAILED.value
        assert any("RuntimeError" in w for w in result.warnings)
        assert result.items == []

    @pytest.mark.asyncio
    async def test_invalid_json_returns_failed(self):
        skill = CurrentCurriculumReviewSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.current_curriculum_review_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "test-key"
            ms.SYNTHESIS_MODEL = "gpt-4o-mini"
            ms.SYNTHESIS_TIMEOUT_S = 5.0
            ms.SYNTHESIS_MAX_TOKENS = 1000
            ms.SYNTHESIS_TEMPERATURE = 0.15

            with patch.object(skill, "_call_openai", return_value="not valid json"):
                result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.status == ReviewStatus.FAILED.value

    def test_parse_empty_items_returns_insufficient(self):
        skill = CurrentCurriculumReviewSkill()
        raw_json = json.dumps({"items": [], "warnings": []})
        result = skill._parse_response(raw_json, [])
        assert result.status == "insufficient_evidence"

    def test_parse_invalid_json_raises(self):
        skill = CurrentCurriculumReviewSkill()
        with pytest.raises(Exception):
            skill._parse_response("not json", [])


# ══════════════════════════════════════════════════════════════════════
# 6. Draft mode: both skills run, 5 others skeleton
# ══════════════════════════════════════════════════════════════════════


class TestDraftModeBothSkills:

    @pytest.mark.asyncio
    async def test_draft_calls_both_skills(self):
        """Draft mode should call EvidenceAnalysisSkill AND CurrentCurriculumReviewSkill."""
        evidence_called = {"count": 0}
        curriculum_called = {"count": 0}

        async def mock_evidence_run(self, *, update_cycle_id, sources, **kwargs):
            evidence_called["count"] += 1
            return EvidenceAnalysisResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_curriculum_run(self, *, update_cycle_id, sources, **kwargs):
            curriculum_called["count"] += 1
            return CurrentCurriculumReviewResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            with patch.object(EvidenceAnalysisSkill, "run", mock_evidence_run):
                with patch.object(CurrentCurriculumReviewSkill, "run", mock_curriculum_run):
                    result = await analyze_update_cycle(
                        AsyncMock(),
                        tenant_id="t1",
                        user_id=1,
                        update_cycle_id="15",
                        analysis_mode="draft",
                    )

        assert evidence_called["count"] == 1
        assert curriculum_called["count"] == 1
        assert set(result.result_payload.keys()) == set(ALL_PAYLOAD_KEYS)

    @pytest.mark.asyncio
    async def test_5_other_sections_still_needs_generation(self):
        """In draft mode, only evidence_summary and evaluation_points may differ."""

        async def mock_evidence_run(self, *, update_cycle_id, sources, **kwargs):
            return EvidenceAnalysisResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_curriculum_run(self, *, update_cycle_id, sources, **kwargs):
            return CurrentCurriculumReviewResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            with patch.object(EvidenceAnalysisSkill, "run", mock_evidence_run):
                with patch.object(CurrentCurriculumReviewSkill, "run", mock_curriculum_run):
                    result = await analyze_update_cycle(
                        AsyncMock(),
                        tenant_id="t1",
                        user_id=1,
                        update_cycle_id="15",
                        analysis_mode="draft",
                    )

        other_keys = [k for k in ALL_PAYLOAD_KEYS
                      if k not in ("evidence_summary", "evaluation_points", "change_proposals")]
        for key in other_keys:
            for item in result.result_payload[key]:
                assert item.status == "needs_generation", f"{key} should be needs_generation"


# ══════════════════════════════════════════════════════════════════════
# 7. Skeleton mode → R4 behavior preserved
# ══════════════════════════════════════════════════════════════════════


class TestSkeletonModePreserved:

    @pytest.mark.asyncio
    async def test_skeleton_mode_all_needs_generation(self):
        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            result = await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="15",
                analysis_mode="skeleton",
            )

        for key, items in result.result_payload.items():
            for item in items:
                assert item.status == "needs_generation", f"{key} should be skeleton"

    @pytest.mark.asyncio
    async def test_skeleton_returns_7_keys(self):
        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            result = await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="15",
                analysis_mode="skeleton",
            )

        assert set(result.result_payload.keys()) == set(ALL_PAYLOAD_KEYS)


# ══════════════════════════════════════════════════════════════════════
# 8. Schema validation
# ══════════════════════════════════════════════════════════════════════


class TestSchemaValidation:

    def test_review_confidence_enum(self):
        assert ReviewConfidence.HIGH.value == "high"
        assert ReviewConfidence.MEDIUM.value == "medium"
        assert ReviewConfidence.LOW.value == "low"
        assert ReviewConfidence.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"

    def test_review_status_enum(self):
        assert ReviewStatus.GENERATED.value == "generated"
        assert ReviewStatus.NEEDS_GENERATION.value == "needs_generation"
        assert ReviewStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert ReviewStatus.FAILED.value == "failed"

    def test_evaluation_point_fields(self):
        src = _make_source()
        point = CurriculumEvaluationPoint(
            aspect="Mục tiêu đào tạo",
            finding="Phù hợp với nhu cầu xã hội",
            recommendation="Bổ sung mục tiêu AI/ML",
            rationale="Xu hướng công nghệ",
            confidence="high",
            sources=[src],
        )
        assert point.aspect == "Mục tiêu đào tạo"
        assert point.confidence == "high"
        assert len(point.sources) == 1

    def test_result_fields(self):
        result = CurrentCurriculumReviewResult(
            status="generated", items=[], warnings=[],
        )
        assert result.task_type == "current_curriculum_review"


# ══════════════════════════════════════════════════════════════════════
# 9. Prompt building
# ══════════════════════════════════════════════════════════════════════


class TestPromptBuilding:

    def test_prompt_includes_program_info(self):
        sources = [_make_source()]
        prompt = _build_user_prompt(
            program_name="CNTT",
            program_code="7480201",
            update_cycle_id="15",
            sources=sources,
        )
        assert "CNTT" in prompt
        assert "7480201" in prompt
        assert "15" in prompt

    def test_prompt_includes_context_text(self):
        sources = [_make_source(quote="130 tín chỉ bắt buộc")]
        prompt = _build_user_prompt(
            program_name=None,
            program_code=None,
            update_cycle_id="15",
            sources=sources,
        )
        assert "130 tín chỉ bắt buộc" in prompt

    def test_prompt_no_sources(self):
        prompt = _build_user_prompt(
            program_name=None, program_code=None,
            update_cycle_id="15", sources=[],
        )
        assert "Không có contexts" in prompt


# ══════════════════════════════════════════════════════════════════════
# 10. Sources preserved in adapter
# ══════════════════════════════════════════════════════════════════════


class TestSourcesPreserved:

    @pytest.mark.asyncio
    async def test_adapter_preserves_sources(self):
        sources = [_make_source(), _make_source(doc_id=43)]

        with patch("app.services.ctdt_skills.current_curriculum_review_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = False
            items = await _run_curriculum_review_skill(
                update_cycle_id="15",
                program_code="7480201",
                program_name="CNTT",
                sources=sources,
            )

        assert len(items) == 1
        assert len(items[0].sources) == 2
        assert items[0].sources[0].ai_document_id == 42
        assert items[0].sources[1].ai_document_id == 43
        assert items[0].task_type == "current_curriculum_review"
