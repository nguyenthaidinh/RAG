"""
Tests for R5.4 — ChangeProposalSkill.

Covers:
1. No contexts → insufficient_evidence
2. LLM disabled → needs_generation, no fabrication
3. LLM enabled + valid JSON → generated, sources mapped
4. source_indices invalid → warning, no fake sources
5. LLM error → failed, analysis doesn't crash
6. Draft mode: 3 skills run, 4 others skeleton
7. Skeleton mode → R4 behavior preserved
8. Schema validation
9. Prompt building
10. Sources preserved in adapter
"""
import json
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.ctdt_analysis_service import (
    ALL_PAYLOAD_KEYS,
    AnalysisCycleResult,
    AnalysisSkeletonItem,
    AnalysisSource,
    AnalysisSourceSummary,
    analyze_update_cycle,
    _run_change_proposal_skill,
)
from app.services.ctdt_skills.change_proposal_skill import (
    ChangeProposalItem,
    ChangeProposalResult,
    ChangeProposalSkill,
    ChangeType,
    ProposalConfidence,
    ProposalPriority,
    ProposalStatus,
    _build_user_prompt,
)
from app.services.ctdt_skills.evidence_analysis_skill import (
    EvidenceAnalysisResult,
    EvidenceAnalysisSkill,
)
from app.services.ctdt_skills.current_curriculum_review_skill import (
    CurrentCurriculumReviewResult,
    CurrentCurriculumReviewSkill,
)
from app.services.ctdt_retrieval_service import (
    CTDTRetrievalContext,
    CTDTRetrievalResult,
    CTDTTaskType,
)


# ── Helpers ──────────────────────────────────────────────────────────

_SKILLED_KEYS = frozenset({"evidence_summary", "evaluation_points", "change_proposals"})


def _make_source(
    *,
    doc_id: int = 42,
    chunk_id: int = 4200001,
    score: float = 0.82,
    quote: str = "Cần bổ sung học phần AI/ML theo xu hướng công nghệ mới.",
    filename: str = "de_xuat_cap_nhat.pdf",
    document_role: str = "change_proposal",
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


def _make_retrieval_context(doc_id=42, chunk_id=4200001, score=0.82):
    return CTDTRetrievalContext(
        ai_document_id=doc_id,
        external_file_id=f"file_{doc_id}",
        filename="de_xuat_cap_nhat.pdf",
        document_role="change_proposal",
        chunk_id=chunk_id,
        chunk_index=chunk_id % 100_000,
        score=score,
        text="Cần bổ sung học phần AI/ML theo xu hướng công nghệ mới.",
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
        skill = ChangeProposalSkill()
        result = await skill.run(update_cycle_id="15", sources=[])
        assert result.status == ProposalStatus.INSUFFICIENT_EVIDENCE.value
        assert result.items == []
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_empty_sources_no_fabrication(self):
        skill = ChangeProposalSkill()
        result = await skill.run(update_cycle_id="15", sources=[])
        assert len(result.items) == 0


# ══════════════════════════════════════════════════════════════════════
# 2. LLM disabled → needs_generation
# ══════════════════════════════════════════════════════════════════════


class TestLLMDisabled:

    @pytest.mark.asyncio
    async def test_synthesis_disabled_returns_needs_generation(self):
        skill = ChangeProposalSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.change_proposal_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = False
            result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.status == ProposalStatus.NEEDS_GENERATION.value
        assert result.items == []
        assert any("SYNTHESIS_ENABLED=false" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_synthesis_disabled_no_fabrication(self):
        skill = ChangeProposalSkill()
        sources = [_make_source(), _make_source(doc_id=43)]

        with patch("app.services.ctdt_skills.change_proposal_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = False
            result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.items == []

    @pytest.mark.asyncio
    async def test_no_api_key_returns_needs_generation(self):
        skill = ChangeProposalSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.change_proposal_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = ""
            result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.status == ProposalStatus.NEEDS_GENERATION.value
        assert any("OPENAI_API_KEY" in w for w in result.warnings)


# ══════════════════════════════════════════════════════════════════════
# 3. LLM enabled + valid JSON → generated
# ══════════════════════════════════════════════════════════════════════


class TestLLMEnabledValidJSON:

    def test_parse_valid_response(self):
        skill = ChangeProposalSkill()
        sources = [_make_source(), _make_source(doc_id=43)]

        raw_json = json.dumps({
            "items": [
                {
                    "target_area": "Chuẩn đầu ra",
                    "change_type": "update",
                    "current_issue": "CĐR chưa phản ánh năng lực AI/ML",
                    "proposed_change": "Bổ sung CĐR về ứng dụng AI cơ bản",
                    "rationale": "78% nhà tuyển dụng yêu cầu kỹ năng AI",
                    "expected_impact": "SV tốt nghiệp đáp ứng tốt hơn",
                    "priority": "high",
                    "confidence": "high",
                    "source_indices": [0, 1],
                },
                {
                    "target_area": "Học phần",
                    "change_type": "add",
                    "current_issue": "Thiếu học phần về AI/ML",
                    "proposed_change": "Thêm 1 học phần nhập môn AI",
                    "rationale": "Xu hướng công nghệ",
                    "expected_impact": "Tăng năng lực AI cho SV",
                    "priority": "medium",
                    "confidence": "medium",
                    "source_indices": [0],
                },
            ],
            "warnings": [],
        })

        result = skill._parse_response(raw_json, sources)
        assert result.status == "generated"
        assert len(result.items) == 2
        assert result.items[0].target_area == "Chuẩn đầu ra"
        assert result.items[0].change_type == "update"
        assert result.items[0].priority == "high"
        assert result.items[0].confidence == "high"
        assert len(result.items[0].sources) == 2
        assert result.items[1].target_area == "Học phần"
        assert result.items[1].change_type == "add"
        assert len(result.items[1].sources) == 1

    def test_parse_sources_mapped_correctly(self):
        skill = ChangeProposalSkill()
        s1 = _make_source(doc_id=42)
        s2 = _make_source(doc_id=43)

        raw_json = json.dumps({
            "items": [{
                "target_area": "Mục tiêu đào tạo",
                "change_type": "clarify",
                "current_issue": "test",
                "proposed_change": "test",
                "rationale": "test",
                "expected_impact": "test",
                "priority": "low",
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
        skill = ChangeProposalSkill()
        sources = [_make_source()]

        raw_json = json.dumps({
            "items": [{
                "target_area": "Test",
                "change_type": "update",
                "current_issue": "test",
                "proposed_change": "test",
                "rationale": "test",
                "expected_impact": "test",
                "priority": "medium",
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
        skill = ChangeProposalSkill()
        sources = [_make_source()]

        raw_json = json.dumps({
            "items": [{
                "target_area": "Test",
                "change_type": "add",
                "current_issue": "test",
                "proposed_change": "test",
                "rationale": "test",
                "expected_impact": "test",
                "priority": "low",
                "confidence": "low",
                "source_indices": [5, 10],
            }],
            "warnings": [],
        })

        result = skill._parse_response(raw_json, sources)
        assert result.items[0].sources == []

    def test_invalid_change_type_defaults_to_other(self):
        skill = ChangeProposalSkill()
        raw_json = json.dumps({
            "items": [{
                "target_area": "Test",
                "change_type": "invalid_type",
                "current_issue": "test",
                "proposed_change": "test",
                "rationale": "test",
                "expected_impact": "test",
                "priority": "medium",
                "confidence": "medium",
                "source_indices": [],
            }],
            "warnings": [],
        })
        result = skill._parse_response(raw_json, [])
        assert result.items[0].change_type == "other"

    def test_invalid_priority_defaults_to_medium(self):
        skill = ChangeProposalSkill()
        raw_json = json.dumps({
            "items": [{
                "target_area": "Test",
                "change_type": "update",
                "current_issue": "test",
                "proposed_change": "test",
                "rationale": "test",
                "expected_impact": "test",
                "priority": "critical",
                "confidence": "medium",
                "source_indices": [],
            }],
            "warnings": [],
        })
        result = skill._parse_response(raw_json, [])
        assert result.items[0].priority == "medium"

    def test_invalid_confidence_defaults_to_medium(self):
        skill = ChangeProposalSkill()
        raw_json = json.dumps({
            "items": [{
                "target_area": "Test",
                "change_type": "update",
                "current_issue": "test",
                "proposed_change": "test",
                "rationale": "test",
                "expected_impact": "test",
                "priority": "medium",
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
        skill = ChangeProposalSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.change_proposal_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "test-key"
            ms.SYNTHESIS_MODEL = "gpt-4o-mini"
            ms.SYNTHESIS_TIMEOUT_S = 5.0
            ms.SYNTHESIS_MAX_TOKENS = 1000
            ms.SYNTHESIS_TEMPERATURE = 0.15

            with patch.object(skill, "_call_openai", side_effect=RuntimeError("API error")):
                result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.status == ProposalStatus.FAILED.value
        assert any("RuntimeError" in w for w in result.warnings)
        assert result.items == []

    @pytest.mark.asyncio
    async def test_invalid_json_returns_failed(self):
        skill = ChangeProposalSkill()
        sources = [_make_source()]

        with patch("app.services.ctdt_skills.change_proposal_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "test-key"
            ms.SYNTHESIS_MODEL = "gpt-4o-mini"
            ms.SYNTHESIS_TIMEOUT_S = 5.0
            ms.SYNTHESIS_MAX_TOKENS = 1000
            ms.SYNTHESIS_TEMPERATURE = 0.15

            with patch.object(skill, "_call_openai", return_value="not valid json"):
                result = await skill.run(update_cycle_id="15", sources=sources)

        assert result.status == ProposalStatus.FAILED.value

    def test_parse_empty_items_returns_insufficient(self):
        skill = ChangeProposalSkill()
        raw_json = json.dumps({"items": [], "warnings": []})
        result = skill._parse_response(raw_json, [])
        assert result.status == "insufficient_evidence"

    def test_parse_invalid_json_raises(self):
        skill = ChangeProposalSkill()
        with pytest.raises(Exception):
            skill._parse_response("not json", [])


# ══════════════════════════════════════════════════════════════════════
# 6. Draft mode: 3 skills run, 4 others skeleton
# ══════════════════════════════════════════════════════════════════════


class TestDraftModeAllThreeSkills:

    @pytest.mark.asyncio
    async def test_draft_calls_all_three_skills(self):
        """Draft mode should call all 3 skills."""
        evidence_called = {"count": 0}
        curriculum_called = {"count": 0}
        proposal_called = {"count": 0}

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

        async def mock_proposal_run(self, *, update_cycle_id, sources, **kwargs):
            proposal_called["count"] += 1
            return ChangeProposalResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            with patch.object(EvidenceAnalysisSkill, "run", mock_evidence_run):
                with patch.object(CurrentCurriculumReviewSkill, "run", mock_curriculum_run):
                    with patch.object(ChangeProposalSkill, "run", mock_proposal_run):
                        result = await analyze_update_cycle(
                            AsyncMock(),
                            tenant_id="t1",
                            user_id=1,
                            update_cycle_id="15",
                            analysis_mode="draft",
                        )

        assert evidence_called["count"] == 1
        assert curriculum_called["count"] == 1
        assert proposal_called["count"] == 1
        assert set(result.result_payload.keys()) == set(ALL_PAYLOAD_KEYS)

    @pytest.mark.asyncio
    async def test_generated_change_proposals_include_payload(self):
        """Generated Mau 06 change proposals should keep draft content in payload."""

        async def mock_evidence_run(self, *, update_cycle_id, sources, **kwargs):
            return EvidenceAnalysisResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_curriculum_run(self, *, update_cycle_id, sources, **kwargs):
            return CurrentCurriculumReviewResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_proposal_run(self, *, update_cycle_id, sources, **kwargs):
            return ChangeProposalResult(
                status="generated",
                items=[
                    ChangeProposalItem(
                        target_area="Chuan dau ra",
                        change_type="update",
                        current_issue="CDR chua phan anh nang luc AI/ML",
                        proposed_change="Bo sung CDR ve ung dung AI co ban",
                        rationale="Minh chung cho thay nhu cau AI tang",
                        expected_impact="Sinh vien dap ung tot hon yeu cau viec lam",
                        priority="high",
                        confidence="high",
                        sources=sources[:1],
                    )
                ],
                warnings=[],
            )

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            with patch.object(EvidenceAnalysisSkill, "run", mock_evidence_run):
                with patch.object(CurrentCurriculumReviewSkill, "run", mock_curriculum_run):
                    with patch.object(ChangeProposalSkill, "run", mock_proposal_run):
                        result = await analyze_update_cycle(
                            AsyncMock(),
                            tenant_id="t1",
                            user_id=1,
                            update_cycle_id="15",
                            analysis_mode="draft",
                        )

        proposals = result.result_payload["change_proposals"]
        assert len(proposals) == 1
        assert proposals[0].payload is not None
        assert proposals[0].payload["proposed_change"] == "Bo sung CDR ve ung dung AI co ban"
        assert set(proposals[0].payload.keys()) == {
            "target_area",
            "change_type",
            "current_issue",
            "proposed_change",
            "rationale",
            "expected_impact",
            "priority",
            "confidence",
        }

    @pytest.mark.asyncio
    async def test_draft_does_not_write_official_program_db(self):
        """Analysis draft must not persist changes to official program tables."""

        async def mock_evidence_run(self, *, update_cycle_id, sources, **kwargs):
            return EvidenceAnalysisResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_curriculum_run(self, *, update_cycle_id, sources, **kwargs):
            return CurrentCurriculumReviewResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_proposal_run(self, *, update_cycle_id, sources, **kwargs):
            return ChangeProposalResult(
                status="generated",
                items=[
                    ChangeProposalItem(
                        target_area="Hoc phan",
                        change_type="add",
                        current_issue="Thieu noi dung AI/ML",
                        proposed_change="Du thao bo sung noi dung AI/ML",
                        rationale="Co minh chung tu tai lieu cap nhat",
                        expected_impact="Tang muc do phu hop voi nhu cau",
                        priority="medium",
                        confidence="medium",
                        sources=sources[:1],
                    )
                ],
                warnings=[],
            )

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        db = AsyncMock()
        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            with patch.object(EvidenceAnalysisSkill, "run", mock_evidence_run):
                with patch.object(CurrentCurriculumReviewSkill, "run", mock_curriculum_run):
                    with patch.object(ChangeProposalSkill, "run", mock_proposal_run):
                        await analyze_update_cycle(
                            db,
                            tenant_id="t1",
                            user_id=1,
                            update_cycle_id="15",
                            analysis_mode="draft",
                        )

        db.add.assert_not_called()
        db.flush.assert_not_called()
        db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_4_other_sections_still_needs_generation(self):
        """In draft mode, only 3 sections use skills; 4 others stay needs_generation."""

        async def mock_evidence_run(self, *, update_cycle_id, sources, **kwargs):
            return EvidenceAnalysisResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_curriculum_run(self, *, update_cycle_id, sources, **kwargs):
            return CurrentCurriculumReviewResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_proposal_run(self, *, update_cycle_id, sources, **kwargs):
            return ChangeProposalResult(
                status="needs_generation", items=[], warnings=[],
            )

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            with patch.object(EvidenceAnalysisSkill, "run", mock_evidence_run):
                with patch.object(CurrentCurriculumReviewSkill, "run", mock_curriculum_run):
                    with patch.object(ChangeProposalSkill, "run", mock_proposal_run):
                        result = await analyze_update_cycle(
                            AsyncMock(),
                            tenant_id="t1",
                            user_id=1,
                            update_cycle_id="15",
                            analysis_mode="draft",
                        )

        other_keys = [k for k in ALL_PAYLOAD_KEYS if k not in _SKILLED_KEYS]
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
                assert item.payload is None

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

    def test_change_type_enum(self):
        assert ChangeType.ADD.value == "add"
        assert ChangeType.UPDATE.value == "update"
        assert ChangeType.REMOVE.value == "remove"
        assert ChangeType.RESTRUCTURE.value == "restructure"
        assert ChangeType.CLARIFY.value == "clarify"
        assert ChangeType.KEEP_WITH_MONITORING.value == "keep_with_monitoring"
        assert ChangeType.OTHER.value == "other"

    def test_priority_enum(self):
        assert ProposalPriority.HIGH.value == "high"
        assert ProposalPriority.MEDIUM.value == "medium"
        assert ProposalPriority.LOW.value == "low"

    def test_confidence_enum(self):
        assert ProposalConfidence.HIGH.value == "high"
        assert ProposalConfidence.MEDIUM.value == "medium"
        assert ProposalConfidence.LOW.value == "low"
        assert ProposalConfidence.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"

    def test_status_enum(self):
        assert ProposalStatus.GENERATED.value == "generated"
        assert ProposalStatus.NEEDS_GENERATION.value == "needs_generation"
        assert ProposalStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"
        assert ProposalStatus.FAILED.value == "failed"

    def test_proposal_item_fields(self):
        src = _make_source()
        item = ChangeProposalItem(
            target_area="Chuẩn đầu ra",
            change_type="update",
            current_issue="CĐR chưa phản ánh AI/ML",
            proposed_change="Bổ sung CĐR về AI",
            rationale="78% NTD yêu cầu",
            expected_impact="SV đáp ứng tốt hơn",
            priority="high",
            confidence="high",
            sources=[src],
        )
        assert item.target_area == "Chuẩn đầu ra"
        assert item.change_type == "update"
        assert item.priority == "high"
        assert len(item.sources) == 1

    def test_result_fields(self):
        result = ChangeProposalResult(
            status="generated", items=[], warnings=[],
        )
        assert result.task_type == "change_proposal"


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
        sources = [_make_source(quote="Bổ sung AI/ML")]
        prompt = _build_user_prompt(
            program_name=None,
            program_code=None,
            update_cycle_id="15",
            sources=sources,
        )
        assert "Bổ sung AI/ML" in prompt

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

        with patch("app.services.ctdt_skills.change_proposal_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = False
            items = await _run_change_proposal_skill(
                update_cycle_id="15",
                program_code="7480201",
                program_name="CNTT",
                sources=sources,
            )

        assert len(items) == 1
        assert len(items[0].sources) == 2
        assert items[0].sources[0].ai_document_id == 42
        assert items[0].sources[1].ai_document_id == 43
        assert items[0].task_type == "change_proposal"

    @pytest.mark.asyncio
    async def test_adapter_empty_result_payload_none(self):
        sources = [_make_source()]

        async def mock_proposal_run(self, *, update_cycle_id, sources, **kwargs):
            return ChangeProposalResult(
                status="needs_generation",
                items=[],
                warnings=[],
            )

        with patch.object(ChangeProposalSkill, "run", mock_proposal_run):
            items = await _run_change_proposal_skill(
                update_cycle_id="15",
                program_code="7480201",
                program_name="CNTT",
                sources=sources,
            )

        assert len(items) == 1
        assert items[0].payload is None
        assert items[0].sources == sources


# =============================================================================
# 11. API payload schema/conversion
# =============================================================================


class TestAPIPayload:

    def test_analyze_item_payload_is_optional(self):
        from app.api.v1.ctdt import AnalyzeSkeletonItem

        item = AnalyzeSkeletonItem(
            status="needs_generation",
            task_type="change_proposal",
            sources=[],
        )

        assert item.payload is None

    def test_analyze_item_accepts_payload(self):
        from app.api.v1.ctdt import AnalyzeSkeletonItem

        item = AnalyzeSkeletonItem(
            status="generated",
            task_type="change_proposal",
            sources=[],
            payload={"proposed_change": "Bo sung CDR ve AI"},
        )

        assert item.payload["proposed_change"] == "Bo sung CDR ve AI"

    @pytest.mark.asyncio
    async def test_api_conversion_preserves_payload(self):
        from app.api.v1.ctdt import (
            AnalyzeUpdateCycleRequest,
            analyze_update_cycle as api_analyze_update_cycle,
        )

        service_result = AnalysisCycleResult(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
            analysis_mode="draft",
            result_payload={
                "change_proposals": [
                    AnalysisSkeletonItem(
                        status="generated",
                        task_type="change_proposal",
                        sources=[_make_source()],
                        payload={"proposed_change": "Bo sung CDR ve AI"},
                    )
                ],
            },
            source_summary=AnalysisSourceSummary(
                contexts_count=1,
                documents_used=[42],
                tasks_executed=["change_proposal"],
                latency_ms=10,
            ),
        )

        async def mock_do_analyze(db, **kwargs):
            return service_result

        with patch(
            "app.services.ctdt_analysis_service.analyze_update_cycle",
            side_effect=mock_do_analyze,
        ):
            response = await api_analyze_update_cycle(
                body=AnalyzeUpdateCycleRequest(
                    update_cycle_id="15",
                    program_code="7480201",
                    analysis_mode="draft",
                ),
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=1),
                query_svc=AsyncMock(),
            )

        item = response.result_payload["change_proposals"][0]
        assert item.payload is not None
        assert item.payload["proposed_change"] == "Bo sung CDR ve AI"
