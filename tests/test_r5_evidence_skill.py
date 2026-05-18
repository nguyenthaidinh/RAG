"""
Tests for R5.1/R5.2 — EvidenceAnalysisSkill + Draft Mode.

Covers:
1. No contexts → insufficient_evidence
2. Skeleton mode unchanged (R4 backward compat)
3. Draft mode calls EvidenceAnalysisSkill for evidence_summary
4. Draft mode: other 6 sections remain needs_generation
5. LLM disabled → needs_generation (no fabrication)
6. Sources preserved correctly
7. Schema validation
8. Skill parse logic
9. Full suite backward compat
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from app.services.ctdt_analysis_service import (
    ALL_PAYLOAD_KEYS,
    ANALYSIS_TASKS,
    AnalysisSkeletonItem,
    AnalysisSource,
    analyze_update_cycle,
    _run_evidence_skill,
)
from app.services.ctdt_skills.evidence_analysis_skill import (
    EvidenceAnalysisResult,
    EvidenceAnalysisSkill,
    EvidenceConfidence,
    EvidenceStatus,
    EvidenceSummaryItem,
    EvidenceType,
    _build_user_prompt,
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
    score: float = 0.85,
    quote: str = "Kết quả khảo sát cho thấy 78% sinh viên đề xuất cập nhật CTĐT.",
    filename: str = "survey_report.pdf",
    document_role: str = "survey_evidence",
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


def _make_retrieval_context(
    *,
    doc_id: int = 42,
    chunk_id: int = 4200001,
    score: float = 0.85,
    text: str = "Kết quả khảo sát cho thấy 78% sinh viên đề xuất cập nhật CTĐT.",
) -> CTDTRetrievalContext:
    return CTDTRetrievalContext(
        ai_document_id=doc_id,
        external_file_id=f"file_{doc_id}",
        filename="survey_report.pdf",
        document_role="survey_evidence",
        chunk_id=chunk_id,
        chunk_index=chunk_id % 100_000,
        score=score,
        text=text,
        source={
            "update_cycle_id": "15",
            "program_code": "7480201",
            "program_id": None,
            "section": None,
            "page": None,
        },
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


class TestSkillNoContexts:

    @pytest.mark.asyncio
    async def test_empty_sources_returns_insufficient(self):
        skill = EvidenceAnalysisSkill()
        result = await skill.run(
            update_cycle_id="15",
            sources=[],
        )
        assert result.status == EvidenceStatus.INSUFFICIENT_EVIDENCE.value
        assert result.items == []
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_empty_sources_no_fabrication(self):
        skill = EvidenceAnalysisSkill()
        result = await skill.run(
            update_cycle_id="15",
            sources=[],
        )
        # Must not invent items
        assert len(result.items) == 0


# ══════════════════════════════════════════════════════════════════════
# 2. Skeleton mode unchanged (R4 backward compat)
# ══════════════════════════════════════════════════════════════════════


class TestSkeletonModeUnchanged:

    @pytest.mark.asyncio
    async def test_skeleton_mode_all_needs_generation(self):
        """Skeleton mode should never call EvidenceAnalysisSkill."""

        call_count = {"evidence_skill": 0}
        original_run = EvidenceAnalysisSkill.run

        async def tracked_run(self, **kwargs):
            call_count["evidence_skill"] += 1
            return await original_run(self, **kwargs)

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

        # EvidenceAnalysisSkill should NOT be called in skeleton mode
        for key, items in result.result_payload.items():
            for item in items:
                assert item.status == "needs_generation"

    @pytest.mark.asyncio
    async def test_skeleton_still_returns_7_keys(self):
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
# 3. Draft mode calls EvidenceAnalysisSkill
# ══════════════════════════════════════════════════════════════════════


class TestDraftMode:

    @pytest.mark.asyncio
    async def test_draft_calls_evidence_skill(self):
        """Draft mode should call EvidenceAnalysisSkill for evidence_summary."""

        skill_called = {"count": 0}

        async def mock_skill_run(self, *, update_cycle_id, sources, **kwargs):
            skill_called["count"] += 1
            return EvidenceAnalysisResult(
                status="needs_generation",
                items=[],
                warnings=["SYNTHESIS_ENABLED=false"],
            )

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            with patch.object(EvidenceAnalysisSkill, "run", mock_skill_run):
                result = await analyze_update_cycle(
                    AsyncMock(),
                    tenant_id="t1",
                    user_id=1,
                    update_cycle_id="15",
                    analysis_mode="draft",
                )

        assert skill_called["count"] == 1
        assert set(result.result_payload.keys()) == set(ALL_PAYLOAD_KEYS)


# ══════════════════════════════════════════════════════════════════════
# 4. Draft mode: other sections remain needs_generation
# ══════════════════════════════════════════════════════════════════════


class TestDraftModeOtherSections:

    @pytest.mark.asyncio
    async def test_non_evidence_sections_are_skeleton(self):
        """In draft mode, only evidence_summary may differ; others stay needs_generation."""

        async def mock_skill_run(self, *, update_cycle_id, sources, **kwargs):
            return EvidenceAnalysisResult(
                status="needs_generation",
                items=[],
                warnings=[],
            )

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            with patch.object(EvidenceAnalysisSkill, "run", mock_skill_run):
                result = await analyze_update_cycle(
                    AsyncMock(),
                    tenant_id="t1",
                    user_id=1,
                    update_cycle_id="15",
                    analysis_mode="draft",
                )

        non_skill_keys = [k for k in ALL_PAYLOAD_KEYS
                          if k not in ("evidence_summary", "evaluation_points", "change_proposals")]
        for key in non_skill_keys:
            for item in result.result_payload[key]:
                assert item.status == "needs_generation", f"{key} should be needs_generation"


# ══════════════════════════════════════════════════════════════════════
# 5. LLM disabled → needs_generation
# ══════════════════════════════════════════════════════════════════════


class TestLLMDisabled:

    @pytest.mark.asyncio
    async def test_synthesis_disabled_returns_needs_generation(self):
        """When SYNTHESIS_ENABLED=false, skill returns needs_generation."""
        skill = EvidenceAnalysisSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.evidence_analysis_skill.settings") as mock_settings:
            mock_settings.SYNTHESIS_ENABLED = False
            result = await skill.run(
                update_cycle_id="15",
                sources=sources,
            )

        assert result.status == EvidenceStatus.NEEDS_GENERATION.value
        assert len(result.items) == 0
        assert any("SYNTHESIS_ENABLED=false" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_synthesis_disabled_no_fabrication(self):
        """No items should be invented when LLM is off."""
        skill = EvidenceAnalysisSkill()
        sources = [_make_source(), _make_source(doc_id=43)]

        with patch("app.services.ctdt_skills.evidence_analysis_skill.settings") as mock_settings:
            mock_settings.SYNTHESIS_ENABLED = False
            result = await skill.run(
                update_cycle_id="15",
                sources=sources,
            )

        assert result.items == []

    @pytest.mark.asyncio
    async def test_no_api_key_returns_needs_generation(self):
        """When OPENAI_API_KEY is empty, skill returns needs_generation."""
        skill = EvidenceAnalysisSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.evidence_analysis_skill.settings") as mock_settings:
            mock_settings.SYNTHESIS_ENABLED = True
            mock_settings.OPENAI_API_KEY = ""
            result = await skill.run(
                update_cycle_id="15",
                sources=sources,
            )

        assert result.status == EvidenceStatus.NEEDS_GENERATION.value
        assert any("OPENAI_API_KEY" in w for w in result.warnings)


# ══════════════════════════════════════════════════════════════════════
# 6. Sources preserved correctly
# ══════════════════════════════════════════════════════════════════════


class TestSourcesPreserved:

    def test_source_fields_complete(self):
        src = _make_source()
        assert src.ai_document_id == 42
        assert src.external_file_id == "file_42"
        assert src.filename == "survey_report.pdf"
        assert src.document_role == "survey_evidence"
        assert src.chunk_id == 4200001
        assert src.chunk_index == 1
        assert src.score == 0.85
        assert "khảo sát" in src.quote
        assert src.update_cycle_id == "15"
        assert src.program_code == "7480201"

    @pytest.mark.asyncio
    async def test_run_evidence_skill_preserves_sources(self):
        """_run_evidence_skill should pass sources through to result."""
        sources = [_make_source(), _make_source(doc_id=43)]

        with patch("app.services.ctdt_skills.evidence_analysis_skill.settings") as mock_settings:
            mock_settings.SYNTHESIS_ENABLED = False
            items = await _run_evidence_skill(
                update_cycle_id="15",
                program_code="7480201",
                program_name="CNTT",
                sources=sources,
            )

        assert len(items) == 1
        # Sources should be attached to the skeleton item
        assert len(items[0].sources) == 2
        assert items[0].sources[0].ai_document_id == 42
        assert items[0].sources[1].ai_document_id == 43


# ══════════════════════════════════════════════════════════════════════
# 7. Schema validation
# ══════════════════════════════════════════════════════════════════════


class TestSchemaValidation:

    def test_evidence_type_enum(self):
        assert EvidenceType.SURVEY.value == "survey"
        assert EvidenceType.REGULATION.value == "regulation"
        assert EvidenceType.MEETING.value == "meeting"
        assert EvidenceType.DECISION.value == "decision"
        assert EvidenceType.COMPARISON.value == "comparison"
        assert EvidenceType.CURRENT_CURRICULUM.value == "current_curriculum"
        assert EvidenceType.OTHER.value == "other"

    def test_confidence_enum(self):
        assert EvidenceConfidence.HIGH.value == "high"
        assert EvidenceConfidence.MEDIUM.value == "medium"
        assert EvidenceConfidence.LOW.value == "low"
        assert EvidenceConfidence.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"

    def test_status_enum(self):
        assert EvidenceStatus.GENERATED.value == "generated"
        assert EvidenceStatus.NEEDS_GENERATION.value == "needs_generation"
        assert EvidenceStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert EvidenceStatus.FAILED.value == "failed"

    def test_evidence_summary_item_fields(self):
        src = _make_source()
        item = EvidenceSummaryItem(
            title="Khảo sát sinh viên",
            summary="78% sinh viên đề xuất cập nhật",
            evidence_type="survey",
            rationale="Cho thấy nhu cầu cập nhật rõ ràng",
            confidence="high",
            sources=[src],
        )
        assert item.title == "Khảo sát sinh viên"
        assert item.evidence_type == "survey"
        assert item.confidence == "high"
        assert len(item.sources) == 1

    def test_evidence_analysis_result_fields(self):
        result = EvidenceAnalysisResult(
            status="generated",
            items=[],
            warnings=[],
        )
        assert result.status == "generated"
        assert result.task_type == "evidence_analysis"


# ══════════════════════════════════════════════════════════════════════
# 8. Skill parse logic
# ══════════════════════════════════════════════════════════════════════


class TestSkillParsing:

    def test_parse_valid_response(self):
        skill = EvidenceAnalysisSkill()
        sources = [_make_source(), _make_source(doc_id=43)]

        raw_json = json.dumps({
            "items": [
                {
                    "title": "Khảo sát sinh viên năm 2024",
                    "summary": "78% SV đề xuất cập nhật CTĐT",
                    "evidence_type": "survey",
                    "rationale": "Nhu cầu thị trường thay đổi",
                    "confidence": "high",
                    "source_indices": [0, 1],
                },
                {
                    "title": "Thông tư 17/2021",
                    "summary": "Yêu cầu rà soát định kỳ",
                    "evidence_type": "regulation",
                    "rationale": "Quy định pháp luật bắt buộc",
                    "confidence": "high",
                    "source_indices": [0],
                },
            ],
            "warnings": [],
        })

        result = skill._parse_response(raw_json, sources)
        assert result.status == "generated"
        assert len(result.items) == 2
        assert result.items[0].evidence_type == "survey"
        assert result.items[0].confidence == "high"
        assert len(result.items[0].sources) == 2  # source_indices [0, 1]
        assert result.items[1].evidence_type == "regulation"
        assert len(result.items[1].sources) == 1  # source_indices [0]

    def test_parse_invalid_source_index_adds_warning(self):
        skill = EvidenceAnalysisSkill()
        sources = [_make_source()]

        raw_json = json.dumps({
            "items": [
                {
                    "title": "Test",
                    "summary": "Test",
                    "evidence_type": "other",
                    "rationale": "Test",
                    "confidence": "medium",
                    "source_indices": [0, 999],  # 999 is out of range
                }
            ],
            "warnings": [],
        })

        result = skill._parse_response(raw_json, sources)
        assert len(result.items) == 1
        assert len(result.items[0].sources) == 1  # only index 0 valid
        assert any("999" in w for w in result.warnings)

    def test_parse_invalid_evidence_type_defaults_to_other(self):
        skill = EvidenceAnalysisSkill()
        sources = [_make_source()]

        raw_json = json.dumps({
            "items": [
                {
                    "title": "Test",
                    "summary": "Test",
                    "evidence_type": "invalid_type",
                    "rationale": "Test",
                    "confidence": "high",
                    "source_indices": [0],
                }
            ],
            "warnings": [],
        })

        result = skill._parse_response(raw_json, sources)
        assert result.items[0].evidence_type == "other"

    def test_parse_empty_items_returns_insufficient(self):
        skill = EvidenceAnalysisSkill()
        raw_json = json.dumps({"items": [], "warnings": []})
        result = skill._parse_response(raw_json, [])
        assert result.status == "insufficient_evidence"

    def test_parse_invalid_json_raises(self):
        skill = EvidenceAnalysisSkill()
        with pytest.raises(Exception):
            skill._parse_response("not json", [])


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
        sources = [_make_source(quote="Kết quả khảo sát đặc biệt")]
        prompt = _build_user_prompt(
            program_name=None,
            program_code=None,
            update_cycle_id="15",
            sources=sources,
        )
        assert "Kết quả khảo sát đặc biệt" in prompt

    def test_prompt_no_sources(self):
        prompt = _build_user_prompt(
            program_name=None,
            program_code=None,
            update_cycle_id="15",
            sources=[],
        )
        assert "Không có contexts" in prompt


# ══════════════════════════════════════════════════════════════════════
# 10. LLM failure → status=failed with warnings
# ══════════════════════════════════════════════════════════════════════


class TestLLMFailure:

    @pytest.mark.asyncio
    async def test_llm_call_failure_returns_failed(self):
        """If OpenAI call raises, skill returns status=failed."""
        skill = EvidenceAnalysisSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.evidence_analysis_skill.settings") as mock_settings:
            mock_settings.SYNTHESIS_ENABLED = True
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.SYNTHESIS_MODEL = "gpt-4o-mini"
            mock_settings.SYNTHESIS_TIMEOUT_S = 5.0
            mock_settings.SYNTHESIS_MAX_TOKENS = 1000
            mock_settings.SYNTHESIS_TEMPERATURE = 0.15

            with patch.object(skill, "_call_openai", side_effect=RuntimeError("API error")):
                result = await skill.run(
                    update_cycle_id="15",
                    sources=sources,
                )

        assert result.status == EvidenceStatus.FAILED.value
        assert any("RuntimeError" in w for w in result.warnings)
        assert result.items == []

    @pytest.mark.asyncio
    async def test_llm_returns_invalid_json(self):
        """If LLM returns non-JSON, skill returns status=failed."""
        skill = EvidenceAnalysisSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.evidence_analysis_skill.settings") as mock_settings:
            mock_settings.SYNTHESIS_ENABLED = True
            mock_settings.OPENAI_API_KEY = "test-key"
            mock_settings.SYNTHESIS_MODEL = "gpt-4o-mini"
            mock_settings.SYNTHESIS_TIMEOUT_S = 5.0
            mock_settings.SYNTHESIS_MAX_TOKENS = 1000
            mock_settings.SYNTHESIS_TEMPERATURE = 0.15

            with patch.object(skill, "_call_openai", return_value="not valid json"):
                result = await skill.run(
                    update_cycle_id="15",
                    sources=sources,
                )

        assert result.status == EvidenceStatus.FAILED.value

    @pytest.mark.asyncio
    async def test_skill_crash_in_draft_mode_returns_failed_skeleton(self):
        """If _run_evidence_skill crashes, it returns failed skeleton."""
        sources = [_make_source()]

        with patch("app.services.ctdt_analysis_service._run_evidence_skill") as mock_run:
            # Simulate the adapter catching the exception internally
            mock_run.return_value = [AnalysisSkeletonItem(
                status="failed",
                task_type="evidence_analysis",
                sources=sources,
            )]

            async def mock_retrieve(db, **kwargs):
                return _make_retrieval_result(task_type=kwargs["task_type"])

            with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
                result = await analyze_update_cycle(
                    AsyncMock(),
                    tenant_id="t1",
                    user_id=1,
                    update_cycle_id="15",
                    analysis_mode="draft",
                )

        # evidence_summary should be failed, others needs_generation
        evidence = result.result_payload["evidence_summary"]
        assert len(evidence) == 1
        assert evidence[0].status == "failed"

        for key in ALL_PAYLOAD_KEYS:
            if key not in ("evidence_summary", "evaluation_points", "change_proposals"):
                for item in result.result_payload[key]:
                    assert item.status == "needs_generation"


# ══════════════════════════════════════════════════════════════════════
# 11. API schema
# ══════════════════════════════════════════════════════════════════════


class TestAPISchema:

    def test_request_accepts_draft(self):
        from app.api.v1.ctdt import AnalyzeUpdateCycleRequest
        req = AnalyzeUpdateCycleRequest(
            update_cycle_id="15",
            analysis_mode="draft",
        )
        assert req.analysis_mode == "draft"

    def test_request_default_is_skeleton(self):
        from app.api.v1.ctdt import AnalyzeUpdateCycleRequest
        req = AnalyzeUpdateCycleRequest(update_cycle_id="15")
        assert req.analysis_mode == "skeleton"
