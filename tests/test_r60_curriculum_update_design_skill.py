"""Tests for R6.0 Curriculum Update Design Skill, Service, and Endpoint."""
from __future__ import annotations

import inspect
import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.services.ctdt_analysis_service import AnalysisSource


# ── Fixture helpers ──────────────────────────────────────────────────


def _make_source(
    *,
    ai_document_id: int = 1,
    document_role: str = "current_curriculum",
    filename: str = "ctdt.pdf",
    score: float = 0.85,
    quote: str = "Nội dung CTĐT hiện hành",
    chunk_index: int = 0,
) -> AnalysisSource:
    return AnalysisSource(
        ai_document_id=ai_document_id,
        external_file_id=f"ext-{ai_document_id}",
        filename=filename,
        document_role=document_role,
        chunk_id=ai_document_id * 100000 + chunk_index,
        chunk_index=chunk_index,
        score=score,
        quote=quote,
        update_cycle_id="15",
        program_code="7480201",
    )


def _make_sources_with_roles() -> list[AnalysisSource]:
    """Create sources covering multiple document roles."""
    return [
        _make_source(ai_document_id=1, document_role="current_curriculum"),
        _make_source(ai_document_id=2, document_role="direction_decision",
                     filename="quyet_dinh.pdf", quote="Quyết định cập nhật"),
        _make_source(ai_document_id=3, document_role="survey_evidence",
                     filename="khaosat.pdf", quote="Kết quả khảo sát"),
    ]


_VALID_LLM_RESPONSE = json.dumps({
    "update_orientation": "Cập nhật theo hướng chuyển đổi số và AI",
    "objective_updates": [
        {
            "current_objective": "Mục tiêu cũ A",
            "proposed_objective": "Mục tiêu mới A+",
            "reason": "Phù hợp xu thế",
            "evidence_refs": [{"source_index": 0}],
            "priority": "high",
            "confidence": "high",
        }
    ],
    "outcome_updates": [
        {
            "current_outcome": "CĐR cũ",
            "proposed_outcome": "CĐR mới",
            "bloom_level": "apply",
            "reason": "Nâng cấp",
            "evidence_refs": [{"source_index": 1}],
            "priority": "medium",
            "confidence": "medium",
        }
    ],
    "curriculum_structure_updates": [
        {
            "area": "major",
            "current_state": "120 TC",
            "proposed_change": "125 TC",
            "reason": "Thêm học phần AI",
            "evidence_refs": [{"source_index": 0}],
            "priority": "high",
            "confidence": "medium",
        }
    ],
    "course_updates": [
        {
            "course_code": None,
            "course_name": "Nhập môn AI",
            "action": "add",
            "current_state": "Chưa có",
            "proposed_change": "Thêm 3TC",
            "reason": "Xu thế ngành",
            "related_outcomes": ["CĐR mới"],
            "evidence_refs": [{"source_index": 2}],
            "priority": "high",
            "confidence": "high",
        }
    ],
    "matrix_update_notes": [
        {
            "outcome": "CĐR mới",
            "course": "Nhập môn AI",
            "suggested_level": "I",
            "reason": "Học phần giới thiệu",
            "evidence_refs": [{"source_index": 0}],
        }
    ],
    "evidence_based_rationale": [
        {
            "claim": "Khảo sát cho thấy 80% nhà tuyển dụng cần AI",
            "evidence_refs": [{"source_index": 2}],
        }
    ],
    "missing_information": [],
    "risks": [
        {
            "risk": "Thiếu giảng viên AI",
            "impact": "high",
            "mitigation": "Tuyển dụng thêm",
        }
    ],
    "next_actions": [
        {
            "action": "Họp bộ môn thảo luận",
            "owner_hint": "bo_mon",
            "priority": "high",
        }
    ],
})


# ══════════════════════════════════════════════════════════════════════
# 1. Skill parse JSON hợp lệ
# ══════════════════════════════════════════════════════════════════════


class TestSkillParseJson:
    def test_skill_parses_valid_json(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        skill = CurriculumUpdateDesignSkill()
        sources = _make_sources_with_roles()
        result = skill._parse_response(_VALID_LLM_RESPONSE, sources)

        assert result.status == "generated"
        assert isinstance(result.payload.update_orientation, str)
        assert isinstance(result.payload.objective_updates, list)
        assert isinstance(result.payload.outcome_updates, list)
        assert isinstance(result.payload.course_updates, list)
        assert result.warnings == []


# ══════════════════════════════════════════════════════════════════════
# 2. Output có update_orientation
# ══════════════════════════════════════════════════════════════════════


class TestOutputHasUpdateOrientation:
    def test_output_has_update_orientation(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        skill = CurriculumUpdateDesignSkill()
        sources = _make_sources_with_roles()
        result = skill._parse_response(_VALID_LLM_RESPONSE, sources)

        assert result.payload.update_orientation == "Cập nhật theo hướng chuyển đổi số và AI"


# ══════════════════════════════════════════════════════════════════════
# 3. Output có objective_updates/outcome_updates/course_updates
# ══════════════════════════════════════════════════════════════════════


class TestOutputHasRequiredArrays:
    def test_output_has_objective_updates(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        skill = CurriculumUpdateDesignSkill()
        sources = _make_sources_with_roles()
        result = skill._parse_response(_VALID_LLM_RESPONSE, sources)

        assert len(result.payload.objective_updates) >= 1
        assert result.payload.objective_updates[0]["proposed_objective"] == "Mục tiêu mới A+"

    def test_output_has_outcome_updates(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        skill = CurriculumUpdateDesignSkill()
        sources = _make_sources_with_roles()
        result = skill._parse_response(_VALID_LLM_RESPONSE, sources)

        assert len(result.payload.outcome_updates) >= 1
        assert result.payload.outcome_updates[0]["bloom_level"] == "apply"

    def test_output_has_course_updates(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        skill = CurriculumUpdateDesignSkill()
        sources = _make_sources_with_roles()
        result = skill._parse_response(_VALID_LLM_RESPONSE, sources)

        assert len(result.payload.course_updates) >= 1
        assert result.payload.course_updates[0]["action"] == "add"
        assert result.payload.course_updates[0]["course_name"] == "Nhập môn AI"


# ══════════════════════════════════════════════════════════════════════
# 4. evidence_refs giữ được source index/document/chunk
# ══════════════════════════════════════════════════════════════════════


class TestEvidenceRefsResolved:
    def test_evidence_refs_resolved_from_sources(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        skill = CurriculumUpdateDesignSkill()
        sources = _make_sources_with_roles()
        result = skill._parse_response(_VALID_LLM_RESPONSE, sources)

        obj_refs = result.payload.objective_updates[0]["evidence_refs"]
        assert len(obj_refs) == 1
        assert obj_refs[0]["source_index"] == 0
        assert obj_refs[0]["ai_document_id"] == 1
        assert obj_refs[0]["document_role"] == "current_curriculum"

    def test_out_of_range_source_index_generates_warning(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        bad_response = json.dumps({
            "update_orientation": "Test",
            "objective_updates": [
                {
                    "current_objective": "A",
                    "proposed_objective": "B",
                    "reason": "C",
                    "evidence_refs": [{"source_index": 999}],
                    "priority": "high",
                    "confidence": "high",
                }
            ],
        })

        skill = CurriculumUpdateDesignSkill()
        sources = [_make_source()]
        result = skill._parse_response(bad_response, sources)

        assert any("999" in w for w in result.warnings)
        assert result.payload.objective_updates[0]["evidence_refs"] == []


# ══════════════════════════════════════════════════════════════════════
# 5. Thiếu current_curriculum thì có missing_information
# ══════════════════════════════════════════════════════════════════════


class TestMissingCurrentCurriculum:
    @pytest.mark.asyncio
    async def test_no_sources_generates_missing_information(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        skill = CurriculumUpdateDesignSkill()
        result = await skill.run(
            update_cycle_id="15",
            program_code="7480201",
            sources=[],
        )

        assert result.status == "insufficient_evidence"
        mi_types = [m["type"] for m in result.payload.missing_information]
        assert "current_curriculum" in mi_types
        assert "direction_decision" in mi_types

    def test_orchestrator_detects_missing_roles(self):
        from app.services.ctdt_curriculum_update_design_service import (
            _detect_missing_roles,
        )
        from app.services.ctdt_retrieval_service import CTDTRetrievalContext

        # Only survey_evidence present — current_curriculum is missing
        contexts = [
            CTDTRetrievalContext(
                ai_document_id=1,
                external_file_id="ext-1",
                filename="survey.pdf",
                document_role="survey_evidence",
                chunk_id=100000,
                chunk_index=0,
                score=0.8,
                text="Khảo sát",
                source={"update_cycle_id": "15"},
            )
        ]

        missing = _detect_missing_roles(contexts)
        mi_types = [m["type"] for m in missing]
        assert "current_curriculum" in mi_types
        assert "direction_decision" in mi_types
        assert "legal_regulation" in mi_types


# ══════════════════════════════════════════════════════════════════════
# 6. Endpoint design-draft trả đúng draft_type
# ══════════════════════════════════════════════════════════════════════


class TestEndpointDraftType:
    @pytest.mark.asyncio
    async def test_endpoint_returns_correct_draft_type(self):
        from app.api.v1.ctdt import (
            CurriculumDesignDraftRequest,
            create_curriculum_design_draft,
        )
        from app.services.ctdt_curriculum_update_design_service import (
            DesignDraftResult,
            DesignDraftSourceSummary,
        )

        mock_result = DesignDraftResult(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
            draft_type="curriculum_update_design",
            draft_id=None,
            draft_saved=False,
            payload={"update_orientation": "Test", "missing_information": []},
            source_summary=DesignDraftSourceSummary(
                contexts_count=3,
                documents_used=[1, 2, 3],
                tasks_executed=["curriculum_update_design"],
                latency_ms=100,
            ),
        )

        async def mock_generate(db, **kwargs):
            return mock_result

        with patch(
            "app.services.ctdt_curriculum_update_design_service.generate_curriculum_update_design_draft",
            side_effect=mock_generate,
        ):
            response = await create_curriculum_design_draft(
                body=CurriculumDesignDraftRequest(
                    update_cycle_id="15",
                    program_code="7480201",
                    program_name="CNTT",
                ),
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        assert response.draft_type == "curriculum_update_design"
        assert response.update_cycle_id == "15"
        assert response.program_code == "7480201"


# ══════════════════════════════════════════════════════════════════════
# 7. save_draft=false không commit DB
# ══════════════════════════════════════════════════════════════════════


class TestSaveDraftFalse:
    @pytest.mark.asyncio
    async def test_save_draft_false_no_db_commit(self):
        from app.api.v1.ctdt import (
            CurriculumDesignDraftRequest,
            create_curriculum_design_draft,
        )
        from app.services.ctdt_curriculum_update_design_service import (
            DesignDraftResult,
            DesignDraftSourceSummary,
        )

        mock_result = DesignDraftResult(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
            draft_type="curriculum_update_design",
            draft_id=None,
            draft_saved=False,
            payload={"update_orientation": "Test"},
            source_summary=DesignDraftSourceSummary(
                contexts_count=0,
                documents_used=[],
                tasks_executed=["curriculum_update_design"],
                latency_ms=50,
            ),
        )

        async def mock_generate(db, **kwargs):
            assert kwargs["save_draft"] is False
            return mock_result

        db = AsyncMock()
        with patch(
            "app.services.ctdt_curriculum_update_design_service.generate_curriculum_update_design_draft",
            side_effect=mock_generate,
        ):
            response = await create_curriculum_design_draft(
                body=CurriculumDesignDraftRequest(
                    update_cycle_id="15",
                    save_draft=False,
                ),
                request=AsyncMock(),
                db=db,
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        assert response.draft_saved is False
        assert response.draft_id is None
        db.add.assert_not_called()
        db.commit.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# 8. save_draft=true lưu draft_type="curriculum_update_design"
# ══════════════════════════════════════════════════════════════════════


class TestSaveDraftTrue:
    @pytest.mark.asyncio
    async def test_save_draft_true_persists_correct_type(self):
        from app.api.v1.ctdt import (
            CurriculumDesignDraftRequest,
            create_curriculum_design_draft,
        )
        from app.services.ctdt_curriculum_update_design_service import (
            DesignDraftResult,
            DesignDraftSourceSummary,
        )

        mock_result = DesignDraftResult(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
            draft_type="curriculum_update_design",
            draft_id=42,
            draft_saved=True,
            payload={"update_orientation": "Test"},
            source_summary=DesignDraftSourceSummary(
                contexts_count=3,
                documents_used=[1, 2, 3],
                tasks_executed=["curriculum_update_design"],
                latency_ms=200,
            ),
        )

        async def mock_generate(db, **kwargs):
            assert kwargs["save_draft"] is True
            return mock_result

        with patch(
            "app.services.ctdt_curriculum_update_design_service.generate_curriculum_update_design_draft",
            side_effect=mock_generate,
        ):
            response = await create_curriculum_design_draft(
                body=CurriculumDesignDraftRequest(
                    update_cycle_id="15",
                    program_code="7480201",
                    save_draft=True,
                ),
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        assert response.draft_saved is True
        assert response.draft_id == 42
        assert response.draft_type == "curriculum_update_design"


# ══════════════════════════════════════════════════════════════════════
# 9. Không ghi Program/ProgramVersion/ProgramVersionRevision
# ══════════════════════════════════════════════════════════════════════


class TestNoProgramWrites:
    def test_skill_has_no_program_model_imports(self):
        """Verify the skill module does not import official program models."""
        import app.services.ctdt_skills.curriculum_update_design_skill as mod

        source = inspect.getsource(mod)
        # Check actual import statements, not docstrings/comments
        import_lines = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        joined = "\n".join(import_lines)
        assert "ProgramVersion" not in joined
        assert "ProgramVersionRevision" not in joined
        assert "from app.db.models.program" not in joined

    def test_service_has_no_program_model_imports(self):
        """Verify the service module does not import official program models."""
        import app.services.ctdt_curriculum_update_design_service as mod

        source = inspect.getsource(mod)
        import_lines = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        joined = "\n".join(import_lines)
        assert "ProgramVersion" not in joined
        assert "ProgramVersionRevision" not in joined
        assert "from app.db.models.program" not in joined


# ══════════════════════════════════════════════════════════════════════
# 10. Retrieval task type registered
# ══════════════════════════════════════════════════════════════════════


class TestRetrievalTaskType:
    def test_curriculum_update_design_task_type_exists(self):
        from app.services.ctdt_retrieval_service import CTDTTaskType

        assert hasattr(CTDTTaskType, "CURRICULUM_UPDATE_DESIGN")
        assert CTDTTaskType.CURRICULUM_UPDATE_DESIGN.value == "curriculum_update_design"

    def test_curriculum_update_design_has_role_policy(self):
        from app.services.ctdt_retrieval_service import (
            CTDTTaskType,
            TASK_ROLE_POLICY,
        )

        roles = TASK_ROLE_POLICY[CTDTTaskType.CURRICULUM_UPDATE_DESIGN]
        assert "direction_decision" in roles
        assert "current_curriculum" in roles
        assert "legal_regulation" in roles
        assert "survey_evidence" in roles

    def test_resolve_document_roles_returns_policy(self):
        from app.services.ctdt_retrieval_service import (
            CTDTTaskType,
            resolve_document_roles,
        )

        roles = resolve_document_roles(CTDTTaskType.CURRICULUM_UPDATE_DESIGN, None)
        assert len(roles) >= 7  # 8 roles in policy


# ══════════════════════════════════════════════════════════════════════
# 11. Enum validation edge cases
# ══════════════════════════════════════════════════════════════════════


class TestEnumValidation:
    def test_invalid_action_defaults_to_keep(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        response = json.dumps({
            "update_orientation": "Test",
            "course_updates": [
                {
                    "course_name": "Test",
                    "action": "invalid_action",
                    "current_state": "",
                    "proposed_change": "",
                    "reason": "",
                    "evidence_refs": [],
                    "priority": "invalid_priority",
                    "confidence": "invalid_confidence",
                }
            ],
        })

        skill = CurriculumUpdateDesignSkill()
        result = skill._parse_response(response, [])

        assert result.payload.course_updates[0]["action"] == "keep"
        assert result.payload.course_updates[0]["priority"] == "medium"
        assert result.payload.course_updates[0]["confidence"] == "medium"

    def test_invalid_area_defaults_to_other(self):
        from app.services.ctdt_skills.curriculum_update_design_skill import (
            CurriculumUpdateDesignSkill,
        )

        response = json.dumps({
            "update_orientation": "Test",
            "curriculum_structure_updates": [
                {
                    "area": "nonexistent",
                    "current_state": "",
                    "proposed_change": "X",
                    "reason": "",
                    "evidence_refs": [],
                    "priority": "high",
                    "confidence": "high",
                }
            ],
        })

        skill = CurriculumUpdateDesignSkill()
        result = skill._parse_response(response, [])

        assert result.payload.curriculum_structure_updates[0]["area"] == "other"


# ══════════════════════════════════════════════════════════════════════
# 12. Endpoint error handling
# ══════════════════════════════════════════════════════════════════════


class TestEndpointErrorHandling:
    @pytest.mark.asyncio
    async def test_service_error_returns_500(self):
        from app.api.v1.ctdt import (
            CurriculumDesignDraftRequest,
            create_curriculum_design_draft,
        )

        async def mock_generate(db, **kwargs):
            raise RuntimeError("LLM exploded")

        with patch(
            "app.services.ctdt_curriculum_update_design_service.generate_curriculum_update_design_draft",
            side_effect=mock_generate,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await create_curriculum_design_draft(
                    body=CurriculumDesignDraftRequest(
                        update_cycle_id="15",
                    ),
                    request=AsyncMock(),
                    db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                    query_svc=AsyncMock(),
                )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "design_draft_error"
