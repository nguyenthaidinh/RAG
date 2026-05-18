"""Tests for R6.1B Objective Update Skill."""
from __future__ import annotations

import inspect
import json
from dataclasses import asdict, field
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services.ctdt_objective_context_service import (
    ContextItem,
    ContextPackSourceSummary,
    ObjectiveUpdateContextPack,
    RoleCoverageItem,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_ctx_item(
    *,
    ai_document_id: int = 1,
    document_role: str = "current_curriculum",
    filename: str = "ctdt.pdf",
    text: str = "Mục tiêu đào tạo kỹ sư CNTT",
    chunk_index: int = 0,
    score: float = 0.85,
) -> ContextItem:
    return ContextItem(
        ai_document_id=ai_document_id,
        external_file_id=f"ext-{ai_document_id}",
        filename=filename,
        document_role=document_role,
        chunk_id=ai_document_id * 100000 + chunk_index,
        chunk_index=chunk_index,
        score=score,
        text=text,
        source={"update_cycle_id": "15", "program_code": "7480201"},
    )


def _make_context_pack(
    *,
    with_contexts: bool = True,
    missing_info: list | None = None,
) -> ObjectiveUpdateContextPack:
    pack = ObjectiveUpdateContextPack(
        update_cycle_id="15",
        program_code="7480201",
        program_name="CNTT",
    )
    if with_contexts:
        pack.current_objective_contexts = [
            _make_ctx_item(ai_document_id=1, document_role="current_curriculum"),
        ]
        pack.direction_contexts = [
            _make_ctx_item(
                ai_document_id=2,
                document_role="direction_decision",
                filename="quyet_dinh.pdf",
                text="Yêu cầu cập nhật CTĐT theo định hướng công nghệ mới",
            ),
        ]
        pack.legal_contexts = [
            _make_ctx_item(
                ai_document_id=3,
                document_role="legal_regulation",
                filename="thong_tu.pdf",
                text="Chuẩn chương trình đào tạo theo Thông tư 17",
            ),
        ]
        for key, roles, count in [
            ("current_objective", ["current_curriculum"], 1),
            ("direction", ["direction_decision"], 1),
            ("legal", ["legal_regulation"], 1),
            ("evidence", ["survey_evidence", "meeting_report"], 0),
            ("comparison", ["comparison_report"], 0),
        ]:
            pack.role_coverage[key] = RoleCoverageItem(
                document_roles=roles,
                context_count=count,
                documents_used=[],
                status="available" if count > 0 else "missing",
                scoped_document_count=count,
                retrieval_status="ok",
            )
    else:
        for key, roles in [
            ("current_objective", ["current_curriculum"]),
            ("direction", ["direction_decision"]),
            ("legal", ["legal_regulation"]),
            ("evidence", ["survey_evidence", "meeting_report"]),
            ("comparison", ["comparison_report"]),
        ]:
            pack.role_coverage[key] = RoleCoverageItem(
                document_roles=roles,
                context_count=0,
                documents_used=[],
                status="missing",
                scoped_document_count=0,
                retrieval_status="ok",
            )

    pack.missing_information = missing_info or []
    pack.source_summary = ContextPackSourceSummary(
        total_contexts=3 if with_contexts else 0,
        documents_used=[1, 2, 3] if with_contexts else [],
        role_groups_retrieved=["current_objective", "direction", "legal"],
        latency_ms=50,
    )
    return pack


def _make_valid_llm_response() -> str:
    """A valid LLM JSON response matching the R6.1B schema."""
    return json.dumps({
        "objective_update_strategy": {
            "summary": "Cập nhật mục tiêu đào tạo theo định hướng CNTT mới",
            "main_drivers": ["school_direction", "legal_regulation"],
            "human_review_required": True,
        },
        "current_objective_analysis": [
            {
                "current_objective": "Đào tạo kỹ sư CNTT",
                "issue": "Chưa phản ánh xu hướng AI/ML",
                "evidence_refs": [{"source_index": 0, "context_group": "current_objective"}],
                "confidence": "high",
            },
        ],
        "proposed_objectives": [
            {
                "objective_type": "general_objective",
                "code": "PO1",
                "is_draft_code": True,
                "update_operation": "revise",
                "mapped_from_current": "Đào tạo kỹ sư CNTT",
                "proposed_content": "Đào tạo nguồn nhân lực CNTT có năng lực ứng dụng AI/ML",
                "rationale": "Theo yêu cầu cập nhật CTĐT",
                "alignment": {
                    "school_direction": "Phù hợp định hướng nhà trường",
                    "legal_regulation": "Đáp ứng TT17",
                },
                "evidence_refs": [
                    {"source_index": 1, "context_group": "direction"},
                ],
                "quality_flags": [],
                "priority": "high",
                "confidence": "high",
            },
        ],
        "alignment_notes": [],
        "objective_quality_review": {
            "overall_assessment": "Bộ mục tiêu phù hợp",
            "strengths": ["Phản ánh xu hướng CNTT"],
            "weaknesses": [],
            "consistency_notes": [],
            "recommendation_for_human_review": ["Cần xác nhận lại với doanh nghiệp"],
        },
        "missing_information": [],
        "risks": [
            {
                "risk": "Thiếu khảo sát doanh nghiệp",
                "impact": "medium",
                "mitigation": "Bổ sung khảo sát trước khi chốt",
            },
        ],
        "next_actions": [
            {
                "action": "Họp hội đồng thẩm định mục tiêu",
                "owner_hint": "hoi_dong",
                "priority": "high",
            },
        ],
    })


# ══════════════════════════════════════════════════════════════════════
# 1. Skill parses valid JSON
# ══════════════════════════════════════════════════════════════════════


class TestSkillParseJson:
    @pytest.mark.asyncio
    async def test_skill_parses_valid_json(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=_make_valid_llm_response(),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as mock_settings:
            mock_settings.SYNTHESIS_ENABLED = True
            mock_settings.OPENAI_API_KEY = "sk-test"

            result = await skill.run(
                update_cycle_id="15",
                program_code="7480201",
                program_name="CNTT",
                context_pack=pack,
            )

        assert result.status == "generated"
        assert result.task_type == "objective_update"


# ══════════════════════════════════════════════════════════════════════
# 2. Output has objective_update_strategy
# ══════════════════════════════════════════════════════════════════════


class TestOutputHasStrategy:
    @pytest.mark.asyncio
    async def test_output_has_objective_update_strategy(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=_make_valid_llm_response(),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        strat = result.payload.objective_update_strategy
        assert "summary" in strat
        assert "main_drivers" in strat
        assert strat["human_review_required"] is True


# ══════════════════════════════════════════════════════════════════════
# 3. Output has proposed_objectives
# ══════════════════════════════════════════════════════════════════════


class TestOutputHasProposed:
    @pytest.mark.asyncio
    async def test_output_has_proposed_objectives(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=_make_valid_llm_response(),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        assert len(result.payload.proposed_objectives) >= 1
        obj = result.payload.proposed_objectives[0]
        assert obj["objective_type"] in ("general_objective", "specific_objective")
        assert "proposed_content" in obj


# ══════════════════════════════════════════════════════════════════════
# 4. Overlap with outcome → quality_flags
# ══════════════════════════════════════════════════════════════════════


class TestOverlapWithOutcome:
    @pytest.mark.asyncio
    async def test_overlaps_with_outcome_in_quality_flags(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        llm_data = json.loads(_make_valid_llm_response())
        llm_data["proposed_objectives"][0]["quality_flags"] = [
            "overlaps_with_outcome",
        ]
        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        flags = result.payload.proposed_objectives[0]["quality_flags"]
        assert "overlaps_with_outcome" in flags


# ══════════════════════════════════════════════════════════════════════
# 5. No evidence → confidence low + missing_evidence flag
# ══════════════════════════════════════════════════════════════════════


class TestNoEvidenceConfidence:
    @pytest.mark.asyncio
    async def test_no_evidence_refs_forces_low_confidence(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        llm_data = json.loads(_make_valid_llm_response())
        llm_data["proposed_objectives"][0]["evidence_refs"] = []
        llm_data["proposed_objectives"][0]["confidence"] = "high"

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        obj = result.payload.proposed_objectives[0]
        assert obj["confidence"] == "low"
        assert "missing_evidence" in obj["quality_flags"]


# ══════════════════════════════════════════════════════════════════════
# 6. AI-proposed code → is_draft_code=true
# ══════════════════════════════════════════════════════════════════════


class TestDraftCode:
    @pytest.mark.asyncio
    async def test_ai_proposed_code_is_draft(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=_make_valid_llm_response(),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        obj = result.payload.proposed_objectives[0]
        assert obj["code"] == "PO1"
        assert obj["is_draft_code"] is True


# ══════════════════════════════════════════════════════════════════════
# 7. Missing info from context pack merged into payload
# ══════════════════════════════════════════════════════════════════════


class TestMissingInfoMerge:
    @pytest.mark.asyncio
    async def test_context_pack_missing_info_in_payload(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        pack = _make_context_pack(
            missing_info=[{"type": "current_objectives", "description": "Thiếu"}],
        )
        skill = ObjectiveUpdateSkill()

        with patch.object(
            skill, "_call_openai", return_value=_make_valid_llm_response(),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        # Skill result has its own missing_information from LLM response
        # but orchestrator merges context_pack missing_information.
        # At skill level, context_pack.missing_information is passed
        # via the insufficient_context path. Let's test via the service.
        assert result.status == "generated"

    @pytest.mark.asyncio
    async def test_orchestrator_merges_context_pack_missing_info(self):
        from app.services.ctdt_objective_update_service import (
            generate_objective_update_draft,
        )
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdatePayload,
            ObjectiveUpdateResult,
            ObjectiveUpdateStatus,
        )

        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.GENERATED,
            payload=ObjectiveUpdatePayload(
                proposed_objectives=[{"code": "PO1"}],
                missing_information=[],  # skill found nothing missing
            ),
            warnings=[],
        )

        pack = _make_context_pack(
            missing_info=[
                {"type": "survey_evidence", "description": "Thiếu khảo sát"},
            ],
        )

        with patch(
            "app.services.ctdt_objective_update_service.build_objective_update_context_pack",
            return_value=pack,
        ), patch(
            "app.services.ctdt_skills.objective_update_skill.ObjectiveUpdateSkill.run",
            return_value=skill_result,
        ):
            result = await generate_objective_update_draft(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        mi_types = [m["type"] for m in result.payload.get("missing_information", [])]
        assert "survey_evidence" in mi_types


# ══════════════════════════════════════════════════════════════════════
# 8. Endpoint returns draft_type="objective_update"
# ══════════════════════════════════════════════════════════════════════


class TestEndpointDraftType:
    @pytest.mark.asyncio
    async def test_endpoint_returns_correct_draft_type(self):
        from app.api.v1.ctdt import (
            ObjectiveDraftRequest,
            generate_objectives_draft,
        )
        from app.services.ctdt_objective_update_service import (
            ObjectiveDraftResult,
            ObjectiveSourceSummary,
        )

        mock_result = ObjectiveDraftResult(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
            draft_type="objective_update",
            draft_id=None,
            draft_saved=False,
            payload={"proposed_objectives": []},
            context_pack_summary={"role_coverage": {}, "missing_information": []},
            source_summary=ObjectiveSourceSummary(
                contexts_count=3,
                documents_used=[1, 2],
                tasks_executed=["objective_update"],
                latency_ms=100,
            ),
            generation_status="generated",
            warnings=[],
        )

        with patch(
            "app.services.ctdt_objective_update_service.generate_objective_update_draft",
            return_value=mock_result,
        ):
            response = await generate_objectives_draft(
                body=ObjectiveDraftRequest(update_cycle_id="15"),
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        assert response.draft_type == "objective_update"
        assert response.update_cycle_id == "15"


# ══════════════════════════════════════════════════════════════════════
# 9. save_draft=false → no DB commit
# ══════════════════════════════════════════════════════════════════════


class TestSaveDraftFalse:
    @pytest.mark.asyncio
    async def test_save_draft_false_no_db_commit(self):
        from app.services.ctdt_objective_update_service import (
            generate_objective_update_draft,
        )
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdatePayload,
            ObjectiveUpdateResult,
            ObjectiveUpdateStatus,
        )

        pack = _make_context_pack()
        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.GENERATED,
            payload=ObjectiveUpdatePayload(),
            warnings=[],
        )

        db = AsyncMock()

        with patch(
            "app.services.ctdt_objective_update_service.build_objective_update_context_pack",
            return_value=pack,
        ), patch(
            "app.services.ctdt_skills.objective_update_skill.ObjectiveUpdateSkill.run",
            return_value=skill_result,
        ):
            result = await generate_objective_update_draft(
                db,
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
                save_draft=False,
            )

        assert result.draft_saved is False
        assert result.draft_id is None
        db.add.assert_not_called()
        db.commit.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# 10. save_draft=true → persists draft_type="objective_update"
# ══════════════════════════════════════════════════════════════════════


class TestSaveDraftTrue:
    @pytest.mark.asyncio
    async def test_save_draft_true_persists_correct_type(self):
        from app.services.ctdt_objective_update_service import (
            generate_objective_update_draft,
        )
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdatePayload,
            ObjectiveUpdateResult,
            ObjectiveUpdateStatus,
        )

        pack = _make_context_pack()
        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.GENERATED,
            payload=ObjectiveUpdatePayload(),
            warnings=[],
        )

        db = AsyncMock()
        # Simulate flush → refresh → commit setting draft.id
        saved_drafts = []

        def capture_add(obj):
            obj.id = 42
            saved_drafts.append(obj)

        db.add = capture_add

        with patch(
            "app.services.ctdt_objective_update_service.build_objective_update_context_pack",
            return_value=pack,
        ), patch(
            "app.services.ctdt_skills.objective_update_skill.ObjectiveUpdateSkill.run",
            return_value=skill_result,
        ), patch(
            "app.db.models.ctdt_analysis_draft.CTDTAnalysisDraft",
        ) as MockDraft:
            mock_instance = MagicMock()
            mock_instance.id = 42
            MockDraft.return_value = mock_instance

            result = await generate_objective_update_draft(
                db,
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
                save_draft=True,
            )

        assert result.draft_saved is True
        # Verify the draft was created with correct type
        call_kwargs = MockDraft.call_args
        assert call_kwargs.kwargs["draft_type"] == "objective_update"
        assert call_kwargs.kwargs["analysis_mode"] == "design"


# ══════════════════════════════════════════════════════════════════════
# 11. No Program model imports
# ══════════════════════════════════════════════════════════════════════


class TestNoProgramWrites:
    def test_skill_has_no_program_model_imports(self):
        import app.services.ctdt_skills.objective_update_skill as mod
        source = inspect.getsource(mod)
        import_lines = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        joined = "\n".join(import_lines)
        assert "ProgramVersion" not in joined
        assert "ProgramVersionRevision" not in joined
        assert "from app.db.models.program" not in joined

    def test_service_has_no_program_model_imports(self):
        import app.services.ctdt_objective_update_service as mod
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
# 12. LLM disabled → needs_generation
# ══════════════════════════════════════════════════════════════════════


class TestLLMDisabled:
    @pytest.mark.asyncio
    async def test_llm_disabled_returns_needs_generation(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = False
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        assert result.status == "needs_generation"
        assert len(result.warnings) >= 1


# ══════════════════════════════════════════════════════════════════════
# 13. No contexts → insufficient_context
# ══════════════════════════════════════════════════════════════════════


class TestNoContexts:
    @pytest.mark.asyncio
    async def test_empty_contexts_returns_insufficient(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack(with_contexts=False)

        result = await skill.run(
            update_cycle_id="15", context_pack=pack,
        )

        assert result.status == "insufficient_context"
        assert "quality_review" in result.payload.objective_quality_review.get(
            "overall_assessment", ""
        ) or "dữ liệu" in result.payload.objective_quality_review.get(
            "overall_assessment", ""
        )


# ══════════════════════════════════════════════════════════════════════
# 14. Endpoint error handling
# ══════════════════════════════════════════════════════════════════════


class TestEndpointErrorHandling:
    @pytest.mark.asyncio
    async def test_service_error_returns_500(self):
        from app.api.v1.ctdt import (
            ObjectiveDraftRequest,
            generate_objectives_draft,
        )

        with patch(
            "app.services.ctdt_objective_update_service.generate_objective_update_draft",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await generate_objectives_draft(
                    body=ObjectiveDraftRequest(update_cycle_id="15"),
                    request=AsyncMock(),
                    db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                    query_svc=AsyncMock(),
                )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "objective_update_error"


# ══════════════════════════════════════════════════════════════════════
# 15. generation_status and warnings in API
# ══════════════════════════════════════════════════════════════════════


class TestGenerationStatusInAPI:
    @pytest.mark.asyncio
    async def test_endpoint_returns_generation_status_and_warnings(self):
        from app.api.v1.ctdt import (
            ObjectiveDraftRequest,
            generate_objectives_draft,
        )
        from app.services.ctdt_objective_update_service import (
            ObjectiveDraftResult,
            ObjectiveSourceSummary,
        )

        mock_result = ObjectiveDraftResult(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
            draft_type="objective_update",
            draft_id=None,
            draft_saved=False,
            payload={},
            context_pack_summary={"role_coverage": {}, "missing_information": []},
            source_summary=ObjectiveSourceSummary(
                contexts_count=0,
                documents_used=[],
                tasks_executed=["objective_update"],
                latency_ms=50,
            ),
            generation_status="needs_generation",
            warnings=["OPENAI_API_KEY chưa được cấu hình."],
        )

        with patch(
            "app.services.ctdt_objective_update_service.generate_objective_update_draft",
            return_value=mock_result,
        ):
            response = await generate_objectives_draft(
                body=ObjectiveDraftRequest(update_cycle_id="15"),
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        assert response.generation_status == "needs_generation"
        assert len(response.warnings) >= 1
        assert "OPENAI_API_KEY" in response.warnings[0]

    @pytest.mark.asyncio
    async def test_no_key_skill_returns_needs_generation(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = None
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        assert result.status == "needs_generation"
        assert len(result.warnings) >= 1

    @pytest.mark.asyncio
    async def test_llm_failure_returns_failed_status(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", side_effect=RuntimeError("timeout"),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        assert result.status == "failed"
        assert len(result.warnings) >= 1


# ══════════════════════════════════════════════════════════════════════
# 16. Evidence refs have source metadata
# ══════════════════════════════════════════════════════════════════════


class TestEvidenceRefsMetadata:
    @pytest.mark.asyncio
    async def test_evidence_refs_include_source_metadata(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        llm_data = json.loads(_make_valid_llm_response())
        # Ensure the ref uses source_index=1 which maps to direction group
        llm_data["proposed_objectives"][0]["evidence_refs"] = [
            {"source_index": 1, "context_group": "direction"},
        ]

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        obj = result.payload.proposed_objectives[0]
        assert len(obj["evidence_refs"]) >= 1
        ref = obj["evidence_refs"][0]
        assert ref["update_cycle_id"] == "15"
        assert ref["program_code"] == "7480201"


# ══════════════════════════════════════════════════════════════════════
# 17. _looks_like_outcome validator
# ══════════════════════════════════════════════════════════════════════


class TestLooksLikeOutcomeValidator:
    def test_general_objective_with_many_action_verbs_detected(self):
        from app.services.ctdt_skills.objective_update_skill import (
            _looks_like_outcome,
        )

        content = (
            "Sinh viên phải có khả năng phân tích, thiết kế và triển khai "
            "các hệ thống phần mềm."
        )
        assert _looks_like_outcome(content, "general_objective") is True

    def test_general_objective_without_action_verbs_not_detected(self):
        from app.services.ctdt_skills.objective_update_skill import (
            _looks_like_outcome,
        )

        content = "Đào tạo nguồn nhân lực CNTT chất lượng cao."
        assert _looks_like_outcome(content, "general_objective") is False

    @pytest.mark.asyncio
    async def test_parser_adds_overlap_flag_for_cdr_like_content(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        llm_data = json.loads(_make_valid_llm_response())
        # Make proposed_content very CĐR-like
        llm_data["proposed_objectives"][0]["proposed_content"] = (
            "Sinh viên có khả năng phân tích, thiết kế, triển khai, "
            "đánh giá và kiểm thử hệ thống."
        )
        llm_data["proposed_objectives"][0]["objective_type"] = "general_objective"
        llm_data["proposed_objectives"][0]["quality_flags"] = []

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        obj = result.payload.proposed_objectives[0]
        assert "overlaps_with_outcome" in obj["quality_flags"]
        assert "needs_human_review" in obj["quality_flags"]
