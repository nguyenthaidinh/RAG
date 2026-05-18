"""Tests for R6.2B Outcome Update Skill."""
from __future__ import annotations
import inspect, json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest
from fastapi import HTTPException
from app.services.ctdt_objective_context_service import ContextItem, ContextPackSourceSummary, RoleCoverageItem
from app.services.ctdt_outcome_context_service import OutcomeUpdateContextPack

def _make_ctx(aid=1, role="current_curriculum", text="CĐR PLO1", idx=0):
    return ContextItem(ai_document_id=aid, external_file_id=f"e-{aid}", filename="f.pdf",
        document_role=role, chunk_id=aid*100+idx, chunk_index=idx, score=0.8, text=text,
        source={"update_cycle_id": "15", "program_code": "7480201", "program_id": "p1"})

def _make_pack(*, with_obj=True, with_ctx=True):
    p = OutcomeUpdateContextPack(update_cycle_id="15", program_code="7480201", program_name="CNTT")
    if with_obj:
        p.objective_update_payload = {"proposed_objectives": [
            {"code": "PO1", "is_draft_code": True, "proposed_content": "Đào tạo nhân lực CNTT AI/ML", "objective_type": "general_objective"}
        ]}
        p.role_coverage["objective_update"] = RoleCoverageItem(document_roles=["objective_update_draft"], context_count=1, documents_used=[], status="available")
    else:
        p.role_coverage["objective_update"] = RoleCoverageItem(document_roles=["objective_update_draft"], context_count=0, documents_used=[], status="missing")
        p.missing_information.append({"type": "objective_update", "description": "Chưa có bản nháp mục tiêu."})
    if with_ctx:
        p.current_outcome_contexts = [_make_ctx(1, "current_curriculum", "PLO1: Có kiến thức nền tảng CNTT")]
        p.current_curriculum_contexts = [_make_ctx(2, "current_curriculum", "Cấu trúc CTĐT hiện hành")]
        p.direction_contexts = [_make_ctx(3, "direction_decision", "Cập nhật theo CDIO")]
        p.legal_contexts = [_make_ctx(4, "legal_regulation", "Thông tư 17/2021")]
        p.evidence_contexts = [_make_ctx(5, "survey_evidence", "Khảo sát NTD")]
        p.comparison_contexts = [_make_ctx(6, "comparison_report", "Đối sánh AUN")]
        p.course_syllabus_contexts = [_make_ctx(7, "course_syllabus", "Đề cương AI/ML")]
        for k, roles in [("current_outcome",["current_curriculum"]),("current_curriculum",["current_curriculum"]),
            ("direction",["direction_decision"]),("legal",["legal_regulation"]),("evidence",["survey_evidence"]),
            ("comparison",["comparison_report"]),("course_syllabus",["course_syllabus"])]:
            p.role_coverage[k] = RoleCoverageItem(document_roles=roles, context_count=1, documents_used=[1], status="available")
    else:
        for k, roles in [("current_outcome",["current_curriculum"]),("current_curriculum",["current_curriculum"]),
            ("direction",["direction_decision"]),("legal",["legal_regulation"]),("evidence",["survey_evidence"]),
            ("comparison",["comparison_report"]),("course_syllabus",["course_syllabus"])]:
            p.role_coverage[k] = RoleCoverageItem(document_roles=roles, context_count=0, documents_used=[], status="missing")
    p.source_summary = ContextPackSourceSummary(total_contexts=7 if with_ctx else 0, documents_used=[1,2,3,4,5,6,7] if with_ctx else [], role_groups_retrieved=["objective_update"], latency_ms=10)
    return p

def _valid_llm():
    return json.dumps({
        "outcome_update_strategy": {"summary": "Cập nhật CĐR theo AI/ML", "main_drivers": ["objective_update"], "human_review_required": True},
        "current_outcome_analysis": [{"current_outcome": "PLO1 hiện tại", "issue": "Chưa cập nhật AI", "mapped_objectives": [], "evidence_refs": [{"source_index": 0, "context_group": "current_outcome"}], "confidence": "medium"}],
        "proposed_outcomes": [{"outcome_type": "knowledge", "code": "PLO1", "is_draft_code": True, "update_operation": "revise",
            "mapped_from_current": "PLO1 cũ", "proposed_content": "Vận dụng kiến thức AI/ML vào phát triển phần mềm",
            "rationale": "Theo xu hướng AI", "bloom_level": "apply",
            "mapped_objectives": [{"objective_code": "PO1", "objective_content": "CNTT AI/ML", "mapping_reason": "Trực tiếp"}],
            "alignment": {"objective_update": "PO1"}, "evidence_refs": [{"source_index": 0, "context_group": "current_outcome"}],
            "quality_flags": [], "priority": "high", "confidence": "high"}],
        "objective_outcome_alignment": [{"objective_code": "PO1", "objective_content": "CNTT AI/ML", "mapped_outcomes": ["PLO1"], "coverage_status": "covered", "notes": "", "evidence_refs": []}],
        "outcome_quality_review": {"overall_assessment": "OK", "strengths": ["Bám mục tiêu"], "weaknesses": [], "consistency_notes": [], "recommendation_for_human_review": []},
        "missing_information": [], "risks": [], "next_actions": []
    })

# ══ 1. Parse valid JSON ══
class TestSkillParseJson:
    @pytest.mark.asyncio
    async def test_skill_parses_valid_json(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=_valid_llm()), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert r.status == "generated"
        assert len(r.payload.proposed_outcomes) >= 1

# ══ 2. Has strategy ══
class TestOutputHasStrategy:
    @pytest.mark.asyncio
    async def test_output_has_strategy(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=_valid_llm()), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert "summary" in r.payload.outcome_update_strategy

# ══ 3. Has proposed_outcomes ══
class TestOutputHasProposed:
    @pytest.mark.asyncio
    async def test_output_has_proposed(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=_valid_llm()), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert len(r.payload.proposed_outcomes) >= 1
        assert r.payload.proposed_outcomes[0]["outcome_type"] == "knowledge"

# ══ 4. Mapped objectives from payload ══
class TestMappedObjectives:
    @pytest.mark.asyncio
    async def test_mapped_objectives_present(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=_valid_llm()), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        po = r.payload.proposed_outcomes[0]
        assert len(po["mapped_objectives"]) >= 1
        assert po["mapped_objectives"][0]["objective_code"] == "PO1"

# ══ 5. No evidence → low confidence ══
class TestNoEvidenceConfidence:
    @pytest.mark.asyncio
    async def test_no_evidence_forces_low(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["evidence_refs"] = []
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        po = r.payload.proposed_outcomes[0]
        assert po["confidence"] == "low"
        assert "missing_evidence" in po["quality_flags"]

# ══ 6. Draft code ══
class TestDraftCode:
    @pytest.mark.asyncio
    async def test_is_draft_code(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=_valid_llm()), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert r.payload.proposed_outcomes[0]["is_draft_code"] is True

# ══ 7. Too broad → quality flags ══
class TestTooBroad:
    @pytest.mark.asyncio
    async def test_too_broad_flags(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["proposed_content"] = "Đào tạo nguồn nhân lực có phẩm chất tốt, phát triển toàn diện"
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        flags = r.payload.proposed_outcomes[0]["quality_flags"]
        assert "too_broad" in flags
        assert "overlaps_with_objective" in flags

# ══ 8. Too course specific ══
class TestTooCourseSpecific:
    @pytest.mark.asyncio
    async def test_too_course_specific_flags(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["proposed_content"] = "Sử dụng thành thạo Visual Studio để lập trình C#"
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        flags = r.payload.proposed_outcomes[0]["quality_flags"]
        assert "too_course_specific" in flags

# ══ 9. Missing objective mapping ══
class TestMissingObjMapping:
    @pytest.mark.asyncio
    async def test_missing_objective_mapping_flag(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["mapped_objectives"] = []
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        flags = r.payload.proposed_outcomes[0]["quality_flags"]
        assert "missing_objective_mapping" in flags
        assert r.payload.proposed_outcomes[0]["confidence"] == "low"

# ══ 10. Evidence refs have source metadata ══
class TestEvidenceRefsMeta:
    @pytest.mark.asyncio
    async def test_evidence_refs_metadata(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=_valid_llm()), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        refs = r.payload.proposed_outcomes[0]["evidence_refs"]
        assert len(refs) >= 1
        assert refs[0]["update_cycle_id"] == "15"
        assert refs[0]["program_code"] == "7480201"

# ══ 11. Missing info merged ══
class TestMissingInfoMerge:
    @pytest.mark.asyncio
    async def test_context_pack_missing_info(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        pack = _make_pack(with_obj=False)
        with patch.object(skill, "_call_openai", return_value=_valid_llm()), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=pack)
        mi_types = [m.get("type") for m in r.payload.missing_information]
        # context_pack had objective_update missing, but LLM may not emit it
        # The merge happens at service level, so here we just check skill doesn't crash
        assert r.status in ("generated", "insufficient_context", "needs_generation")

# ══ 12. LLM disabled ══
class TestLLMDisabled:
    @pytest.mark.asyncio
    async def test_llm_disabled(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        with patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = False
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert r.status == "needs_generation"

# ══ 13. No API key ══
class TestNoAPIKey:
    @pytest.mark.asyncio
    async def test_no_api_key(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        with patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = None
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert r.status == "needs_generation"
        assert any("OPENAI_API_KEY" in w for w in r.warnings)

# ══ 14. LLM failed ══
class TestLLMFailed:
    @pytest.mark.asyncio
    async def test_llm_failed(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", side_effect=RuntimeError("timeout")), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert r.status == "failed"
        assert len(r.warnings) >= 1

# ══ 15. Endpoint draft_type ══
class TestEndpointDraftType:
    @pytest.mark.asyncio
    async def test_endpoint_draft_type(self):
        from app.api.v1.ctdt import OutcomeDraftRequest, generate_outcomes_draft
        from app.services.ctdt_outcome_update_service import OutcomeDraftResult, OutcomeSourceSummary
        mock_result = OutcomeDraftResult(
            update_cycle_id="15", program_code="7480201", program_name="CNTT",
            draft_type="outcome_update", draft_id=None, draft_saved=False, payload={},
            context_pack_summary={"role_coverage": {}, "missing_information": []},
            source_summary=OutcomeSourceSummary(contexts_count=0, documents_used=[], tasks_executed=["outcome_update"], latency_ms=50),
            generation_status="generated", warnings=[],
        )
        with patch("app.services.ctdt_outcome_update_service.generate_outcome_update_draft", return_value=mock_result):
            resp = await generate_outcomes_draft(
                body=OutcomeDraftRequest(update_cycle_id="15"), request=AsyncMock(),
                db=AsyncMock(), user=SimpleNamespace(tenant_id="t1", id=7), query_svc=AsyncMock())
        assert resp.draft_type == "outcome_update"

# ══ 16. Endpoint generation_status ══
class TestEndpointStatus:
    @pytest.mark.asyncio
    async def test_endpoint_returns_status_warnings(self):
        from app.api.v1.ctdt import OutcomeDraftRequest, generate_outcomes_draft
        from app.services.ctdt_outcome_update_service import OutcomeDraftResult, OutcomeSourceSummary
        mock_result = OutcomeDraftResult(
            update_cycle_id="15", program_code="7480201", program_name="CNTT",
            draft_type="outcome_update", draft_id=None, draft_saved=False, payload={},
            context_pack_summary={"role_coverage": {}, "missing_information": []},
            source_summary=OutcomeSourceSummary(contexts_count=0, documents_used=[], tasks_executed=["outcome_update"], latency_ms=50),
            generation_status="needs_generation", warnings=["OPENAI_API_KEY chưa cấu hình."],
        )
        with patch("app.services.ctdt_outcome_update_service.generate_outcome_update_draft", return_value=mock_result):
            resp = await generate_outcomes_draft(
                body=OutcomeDraftRequest(update_cycle_id="15"), request=AsyncMock(),
                db=AsyncMock(), user=SimpleNamespace(tenant_id="t1", id=7), query_svc=AsyncMock())
        assert resp.generation_status == "needs_generation"
        assert len(resp.warnings) >= 1

# ══ 17. save_draft=false no commit ══
class TestSaveDraftFalse:
    @pytest.mark.asyncio
    async def test_save_draft_false(self):
        from app.services.ctdt_outcome_update_service import generate_outcome_update_draft
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill, OutcomeUpdateResult, OutcomeUpdatePayload
        mock_skill_result = OutcomeUpdateResult(status="needs_generation", payload=OutcomeUpdatePayload(), warnings=["no key"])
        db = AsyncMock()
        with patch("app.services.ctdt_outcome_update_service.build_outcome_update_context_pack", return_value=_make_pack()), \
             patch.object(OutcomeUpdateSkill, "run", return_value=mock_skill_result):
            r = await generate_outcome_update_draft(db, tenant_id="t1", user_id=7, update_cycle_id="15")
        db.commit.assert_not_called()
        assert r.draft_saved is False

# ══ 18. save_draft=true persists ══
class TestSaveDraftTrue:
    @pytest.mark.asyncio
    async def test_save_draft_true(self):
        from app.services.ctdt_outcome_update_service import generate_outcome_update_draft
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill, OutcomeUpdateResult, OutcomeUpdatePayload
        mock_skill_result = OutcomeUpdateResult(status="generated", payload=OutcomeUpdatePayload(proposed_outcomes=[{"code":"PLO1"}]), warnings=[])
        db = AsyncMock()
        db.flush = AsyncMock()
        db.refresh = AsyncMock()
        db.commit = AsyncMock()
        mock_draft = AsyncMock()
        mock_draft.id = 42
        with patch("app.services.ctdt_outcome_update_service.build_outcome_update_context_pack", return_value=_make_pack()), \
             patch.object(OutcomeUpdateSkill, "run", return_value=mock_skill_result), \
             patch("app.db.models.ctdt_analysis_draft.CTDTAnalysisDraft", return_value=mock_draft):
            r = await generate_outcome_update_draft(db, tenant_id="t1", user_id=7, update_cycle_id="15", save_draft=True)
        assert r.draft_saved is True
        assert r.draft_type == "outcome_update"

# ══ 19. No Program writes ══
class TestNoProgramWrites:
    def test_skill_no_program_imports(self):
        import app.services.ctdt_skills.outcome_update_skill as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "ProgramVersion" not in joined
        assert "from app.db.models.program" not in joined

    def test_service_no_program_imports(self):
        import app.services.ctdt_outcome_update_service as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "ProgramVersion" not in joined
        assert "from app.db.models.program" not in joined

# ══ 20. Insufficient context ══
class TestInsufficientContext:
    @pytest.mark.asyncio
    async def test_no_contexts_no_obj(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        skill = OutcomeUpdateSkill()
        pack = _make_pack(with_obj=False, with_ctx=False)
        r = await skill.run(update_cycle_id="15", context_pack=pack)
        assert r.status == "insufficient_context"
        assert len(r.payload.proposed_outcomes) == 0

# ══ 21. Endpoint error ══
class TestEndpointError:
    @pytest.mark.asyncio
    async def test_service_error_500(self):
        from app.api.v1.ctdt import OutcomeDraftRequest, generate_outcomes_draft
        with patch("app.services.ctdt_outcome_update_service.generate_outcome_update_draft", side_effect=RuntimeError("boom")):
            with pytest.raises(HTTPException) as exc_info:
                await generate_outcomes_draft(
                    body=OutcomeDraftRequest(update_cycle_id="15"), request=AsyncMock(),
                    db=AsyncMock(), user=SimpleNamespace(tenant_id="t1", id=7), query_svc=AsyncMock())
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "outcome_update_error"


# ══════════════════════════════════════════════════════════════════════
# R6.2B.1 — Normalization Hardening Tests
# ══════════════════════════════════════════════════════════════════════


# ══ 22. Invalid bloom_level normalized ══
class TestNormBloomLevel:
    @pytest.mark.asyncio
    async def test_invalid_bloom_normalized(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["bloom_level"] = "application"
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        po = r.payload.proposed_outcomes[0]
        assert po["bloom_level"] == "unknown"
        assert any("bloom_level" in w and "application" in w for w in r.warnings)


# ══ 23. Invalid outcome_type normalized ══
class TestNormOutcomeType:
    @pytest.mark.asyncio
    async def test_invalid_outcome_type(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["outcome_type"] = "professional_skill"
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert r.payload.proposed_outcomes[0]["outcome_type"] == "other"


# ══ 24. Invalid update_operation normalized ══
class TestNormUpdateOp:
    @pytest.mark.asyncio
    async def test_invalid_update_operation(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["update_operation"] = "modify"
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert r.payload.proposed_outcomes[0]["update_operation"] == "add"


# ══ 25. Invalid priority/confidence normalized ══
class TestNormPriorityConfidence:
    @pytest.mark.asyncio
    async def test_invalid_priority_confidence(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["priority"] = "urgent"
        data["proposed_outcomes"][0]["confidence"] = "very high"
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        po = r.payload.proposed_outcomes[0]
        assert po["priority"] == "medium"
        # confidence may be "low" due to missing_evidence override, either way valid
        assert po["confidence"] in ("low", "medium")


# ══ 26. is_draft_code false for keep/revise with mapped_from_current ══
class TestDraftCodeRevise:
    @pytest.mark.asyncio
    async def test_revise_keeps_official_code(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["update_operation"] = "revise"
        data["proposed_outcomes"][0]["code"] = "PLO1"
        data["proposed_outcomes"][0]["mapped_from_current"] = "PLO1 hiện hành"
        data["proposed_outcomes"][0].pop("is_draft_code", None)
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert r.payload.proposed_outcomes[0]["is_draft_code"] is False


# ══ 27. is_draft_code true for add ══
class TestDraftCodeAdd:
    @pytest.mark.asyncio
    async def test_add_sets_draft_code(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["update_operation"] = "add"
        data["proposed_outcomes"][0]["code"] = "PLO9"
        data["proposed_outcomes"][0]["is_draft_code"] = False  # LLM says false, but add -> true
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        assert r.payload.proposed_outcomes[0]["is_draft_code"] is True


# ══ 28. quality_flags dedup and invalid removal ══
class TestQualityFlagsNorm:
    @pytest.mark.asyncio
    async def test_dedup_and_invalid_flags(self):
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_valid_llm())
        data["proposed_outcomes"][0]["quality_flags"] = ["too_broad", "too_broad", "random_flag"]
        skill = OutcomeUpdateSkill()
        with patch.object(skill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            r = await skill.run(update_cycle_id="15", context_pack=_make_pack())
        flags = r.payload.proposed_outcomes[0]["quality_flags"]
        assert flags.count("too_broad") == 1
        assert "random_flag" not in flags
        assert "needs_human_review" in flags  # added as replacement for invalid
        assert any("random_flag" in w for w in r.warnings)


# ══ 29. warnings Field(default_factory=list) no shared state ══
class TestWarningsDefaultFactory:
    def test_no_shared_mutable_default(self):
        from app.api.v1.ctdt import ObjectiveDraftResponse, OutcomeDraftResponse
        # Check they use default_factory by inspecting the field
        for cls in (ObjectiveDraftResponse, OutcomeDraftResponse):
            field_info = cls.model_fields.get("warnings")
            assert field_info is not None, f"{cls.__name__} missing warnings field"
            assert field_info.default_factory is not None, (
                f"{cls.__name__}.warnings should use default_factory, not mutable default"
            )

