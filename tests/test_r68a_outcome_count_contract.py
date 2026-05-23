"""R6.8A-PATCH-1 tests — allocation parity, quality gate, snapshot, persistence."""
from __future__ import annotations
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch, MagicMock
import pytest
from fastapi import HTTPException
from app.services.ctdt_objective_context_service import ContextItem, ContextPackSourceSummary, RoleCoverageItem
from app.services.ctdt_outcome_context_service import OutcomeUpdateContextPack
from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateStatus

def _make_pack(*, with_obj=True):
    p = OutcomeUpdateContextPack(update_cycle_id="15", program_code="7480201", program_name="CNTT")
    if with_obj:
        p.objective_update_payload = {"proposed_objectives": [
            {"code": "M1", "proposed_content": "Đào tạo nhân lực CNTT", "is_draft_code": False}]}
        p.objective_source = "laravel_approved_objectives"
        p.role_coverage["objective_update"] = RoleCoverageItem(
            document_roles=["laravel_approved"], context_count=1, documents_used=[], status="available")
    else:
        p.objective_source = "none"
        p.role_coverage["objective_update"] = RoleCoverageItem(
            document_roles=["objective_update_draft"], context_count=0, documents_used=[], status="missing")
    p.current_outcome_contexts = [ContextItem(ai_document_id=1, external_file_id="e-1", filename="f.pdf",
        document_role="current_curriculum", chunk_id=100, chunk_index=0, score=0.8, text="CĐR hiện hành",
        source={"update_cycle_id": "15", "program_code": "7480201", "program_id": "p1"})]
    for k, roles in [("current_outcome", ["current_curriculum"]), ("current_curriculum", ["current_curriculum"]),
        ("direction", ["direction_decision"]), ("legal", ["legal_regulation"]), ("evidence", ["survey_evidence"]),
        ("comparison", ["comparison_report"]), ("course_syllabus", ["course_syllabus"])]:
        p.role_coverage[k] = RoleCoverageItem(document_roles=roles, context_count=1, documents_used=[1], status="available")
    p.source_summary = ContextPackSourceSummary(total_contexts=1, documents_used=[1],
        role_groups_retrieved=["objective_update"], latency_ms=10)
    return p

_TABLE = {
    6: {"knowledge": 2, "skills": 3, "autonomy_responsibility": 1},
    7: {"knowledge": 3, "skills": 3, "autonomy_responsibility": 1},
    8: {"knowledge": 3, "skills": 4, "autonomy_responsibility": 1},
    9: {"knowledge": 3, "skills": 4, "autonomy_responsibility": 2},
    10: {"knowledge": 4, "skills": 4, "autonomy_responsibility": 2},
    11: {"knowledge": 4, "skills": 5, "autonomy_responsibility": 2},
    12: {"knowledge": 4, "skills": 6, "autonomy_responsibility": 2},
    13: {"knowledge": 5, "skills": 6, "autonomy_responsibility": 2},
    14: {"knowledge": 5, "skills": 7, "autonomy_responsibility": 2},
    15: {"knowledge": 6, "skills": 7, "autonomy_responsibility": 2},
}

def _llm_response(n=10, allocation=None):
    from app.services.ctdt_skills.outcome_update_skill import _compute_default_allocation
    alloc = allocation or _compute_default_allocation(n)
    outcomes, idx = [], 1
    for group in ("knowledge", "skills", "autonomy_responsibility"):
        for _ in range(alloc.get(group, 0)):
            outcomes.append({"outcome_type": group, "code": f"C{idx}", "is_draft_code": True,
                "update_operation": "add", "mapped_from_current": "",
                "proposed_content": (
                    f"CĐR {group} #{idx} mô tả năng lực người học đạt được sau tốt nghiệp."
                ), "rationale": "Test", "bloom_level": "apply",
                "mapped_objectives": [{"objective_code": "M1", "objective_content": "CNTT", "mapping_reason": "ok"}],
                "alignment": {}, "evidence_refs": [{"source_index": 0, "context_group": "current_outcome"}],
                "quality_flags": [], "priority": "medium", "confidence": "high"})
            idx += 1
    return json.dumps({"outcome_update_strategy": {"summary": "T", "main_drivers": ["objective_update"], "human_review_required": True},
        "current_outcome_analysis": [], "proposed_outcomes": outcomes, "objective_outcome_alignment": [],
        "outcome_quality_review": {"overall_assessment": "OK", "strengths": [], "weaknesses": [], "consistency_notes": [], "recommendation_for_human_review": []},
        "missing_information": [], "risks": [], "next_actions": []})

async def _run_skill(llm_resp, count=10, alloc=None):
    from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
    skill = OutcomeUpdateSkill()
    with patch.object(skill, "_call_openai", return_value=llm_resp), \
         patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
        ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
        return await skill.run(update_cycle_id="15", context_pack=_make_pack(), outcome_count=count, group_allocation=alloc)

async def _run_service(llm_resp, count=10, pack=None, save_draft=False, db=None):
    from app.services.ctdt_outcome_update_service import generate_outcome_update_draft
    from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
    pack = pack or _make_pack()
    db = db or AsyncMock()
    with patch("app.services.ctdt_outcome_update_service.build_outcome_update_context_pack",
                return_value=pack), \
         patch.object(OutcomeUpdateSkill, "_call_openai", return_value=llm_resp), \
         patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms:
        ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
        return await generate_outcome_update_draft(
            db, tenant_id="t1", user_id=7, update_cycle_id="15",
            outcome_count=count, save_draft=save_draft)

# ══ FIX 1: Exact allocation parity ══
class TestAllocationTable:
    @pytest.mark.parametrize("n", range(6, 16))
    def test_full_table(self, n):
        from app.services.ctdt_skills.outcome_update_skill import _compute_default_allocation
        assert _compute_default_allocation(n) == _TABLE[n]

    def test_specific_6(self):
        from app.services.ctdt_skills.outcome_update_skill import _compute_default_allocation
        assert _compute_default_allocation(6) == {"knowledge": 2, "skills": 3, "autonomy_responsibility": 1}

    def test_specific_8(self):
        from app.services.ctdt_skills.outcome_update_skill import _compute_default_allocation
        assert _compute_default_allocation(8) == {"knowledge": 3, "skills": 4, "autonomy_responsibility": 1}

    def test_specific_12(self):
        from app.services.ctdt_skills.outcome_update_skill import _compute_default_allocation
        assert _compute_default_allocation(12) == {"knowledge": 4, "skills": 6, "autonomy_responsibility": 2}

    def test_specific_15(self):
        from app.services.ctdt_skills.outcome_update_skill import _compute_default_allocation
        assert _compute_default_allocation(15) == {"knowledge": 6, "skills": 7, "autonomy_responsibility": 2}

    def test_clamp_5(self):
        from app.services.ctdt_skills.outcome_update_skill import _compute_default_allocation
        assert _compute_default_allocation(5) == _TABLE[6]

    def test_clamp_16(self):
        from app.services.ctdt_skills.outcome_update_skill import _compute_default_allocation
        assert _compute_default_allocation(16) == _TABLE[15]

# ══ FIX 2: No placeholder ══
class TestNoPlaceholder:
    @pytest.mark.asyncio
    async def test_deficit_no_placeholder(self):
        r = await _run_skill(_llm_response(6), count=10)
        for po in r.payload.proposed_outcomes:
            assert "[Cần bổ sung" not in po.get("proposed_content", "")
        assert r.status != "generated" or len(r.payload.proposed_outcomes) < 10

    @pytest.mark.asyncio
    async def test_deficit_quality_failed(self):
        r = await _run_skill(_llm_response(6), count=10)
        assert r.status in ("failed", OutcomeUpdateStatus.FAILED)
        assert any("6/10" in w for w in r.warnings)

# ══ FIX 3: Group normalize by position ══
class TestGroupNormalize:
    @pytest.mark.asyncio
    async def test_all_knowledge_normalized(self):
        data = json.loads(_llm_response(10))
        for po in data["proposed_outcomes"]:
            po["outcome_type"] = "knowledge"
        r = await _run_skill(json.dumps(data), count=10)
        assert len(r.payload.proposed_outcomes) == 10
        types = [po["outcome_type"] for po in r.payload.proposed_outcomes]
        assert types[:4] == ["knowledge"] * 4
        assert types[4:8] == ["skills"] * 4
        assert types[8:] == ["autonomy_responsibility"] * 2
        assert any("chuẩn hóa nhóm" in w for w in r.warnings)
        for po in r.payload.proposed_outcomes:
            assert po["proposed_content"].startswith("CĐR")

# ══ FIX 4: Snapshot contract ══
class TestSnapshotContract:
    @pytest.mark.asyncio
    async def test_valid_snapshot(self):
        from app.services.ctdt_outcome_context_service import build_outcome_update_context_pack
        snap = {"source": "laravel_program_objective_draft", "is_completed": True,
            "general_objective": "Đào tạo nhân lực CNTT chất lượng cao",
            "specific_objectives": [{"code": "M1", "group": "knowledge", "text": "Kiến thức nền tảng"},
                {"code": "M2", "group": "skills_attitude", "text": "Kỹ năng lập trình"}]}
        with patch("app.services.ctdt_outcome_context_service.ctdt_retrieve",
                    side_effect=lambda *a, **kw: _empty_ret(kw)), \
             patch("app.services.ctdt_outcome_context_service._load_latest_objective_draft") as ml:
            pack = await build_outcome_update_context_pack(
                AsyncMock(), tenant_id="t1", user_id=7, update_cycle_id="15",
                approved_objective_snapshot=snap)
        ml.assert_not_called()
        assert pack.objective_source == "laravel_approved_objectives"
        assert len(pack.objective_update_payload["proposed_objectives"]) == 2
        assert pack.objective_update_payload["_general_objective"] == "Đào tạo nhân lực CNTT chất lượng cao"

    @pytest.mark.asyncio
    async def test_incomplete_snapshot_fallback(self):
        from app.services.ctdt_outcome_context_service import build_outcome_update_context_pack
        snap = {"is_completed": False, "specific_objectives": [{"code": "M1", "text": "x"}]}
        with patch("app.services.ctdt_outcome_context_service.ctdt_retrieve",
                    side_effect=lambda *a, **kw: _empty_ret(kw)), \
             patch("app.services.ctdt_outcome_context_service._load_latest_objective_draft", return_value=None):
            pack = await build_outcome_update_context_pack(
                AsyncMock(), tenant_id="t1", user_id=7, update_cycle_id="15",
                approved_objective_snapshot=snap)
        assert pack.objective_source == "none"
        assert any("snapshot" in mi.get("description", "").lower() or "snapshot" in mi.get("type", "").lower()
                    for mi in pack.missing_information)

    @pytest.mark.asyncio
    async def test_legacy_list_still_works(self):
        from app.services.ctdt_outcome_context_service import build_outcome_update_context_pack
        legacy = [{"code": "M1", "content": "Mục tiêu 1"}]
        with patch("app.services.ctdt_outcome_context_service.ctdt_retrieve",
                    side_effect=lambda *a, **kw: _empty_ret(kw)), \
             patch("app.services.ctdt_outcome_context_service._load_latest_objective_draft") as ml:
            pack = await build_outcome_update_context_pack(
                AsyncMock(), tenant_id="t1", user_id=7, update_cycle_id="15",
                approved_objectives=legacy)
        ml.assert_not_called()
        assert pack.objective_source == "legacy_laravel_approved_objectives"

# ══ FIX 5: Prompt M/C codes ══
class TestFlatObjectiveDraftFallback:
    @pytest.mark.asyncio
    async def test_flat_specific_objectives_build_prompt_m_codes(self):
        from app.services.ctdt_outcome_context_service import build_outcome_update_context_pack
        from app.services.ctdt_skills.outcome_update_skill import _build_user_prompt
        draft = {"_flat": {
            "general_objective_text": "Mục tiêu chung",
            "specific_objectives": [
                {"code": "M1", "group": "knowledge", "text": "Kiến thức nền tảng"},
                {"code": "M2", "group": "skills", "text": "Kỹ năng thực hành"},
            ],
        }}
        with patch("app.services.ctdt_outcome_context_service.ctdt_retrieve",
                    side_effect=lambda *a, **kw: _empty_ret(kw)), \
             patch("app.services.ctdt_outcome_context_service._load_latest_objective_draft",
                    return_value=draft):
            pack = await build_outcome_update_context_pack(
                AsyncMock(), tenant_id="t1", user_id=7, update_cycle_id="15")
        prompt, _ = _build_user_prompt(
            program_name="CNTT", program_code="7480201",
            update_cycle_id="15", context_pack=pack)
        assert pack.objective_source == "rag_latest_objective_draft_fallback"
        assert [o["code"] for o in pack.objective_update_payload["proposed_objectives"]] == ["M1", "M2"]
        assert "M1" in prompt and "M2" in prompt

    @pytest.mark.asyncio
    async def test_flat_specific_objective_texts_assigns_m_codes_and_warns(self):
        from app.services.ctdt_outcome_context_service import build_outcome_update_context_pack
        draft = {"_flat": {
            "general_objective_text": "Mục tiêu chung",
            "specific_objective_texts": ["Kiến thức nền tảng", "Kỹ năng thực hành"],
        }}
        with patch("app.services.ctdt_outcome_context_service.ctdt_retrieve",
                    side_effect=lambda *a, **kw: _empty_ret(kw)), \
             patch("app.services.ctdt_outcome_context_service._load_latest_objective_draft",
                    return_value=draft):
            pack = await build_outcome_update_context_pack(
                AsyncMock(), tenant_id="t1", user_id=7, update_cycle_id="15")
        objectives = pack.objective_update_payload["proposed_objectives"]
        assert [o["code"] for o in objectives] == ["M1", "M2"]
        assert any("M1..Mn" in w for w in pack.objective_warnings)
        assert any(mi["type"] == "objective_draft_flat_adapter" for mi in pack.missing_information)

class TestPromptCodes:
    def test_prompt_has_m_codes(self):
        from app.services.ctdt_skills.outcome_update_skill import _build_user_prompt
        pack = _make_pack()
        prompt, _ = _build_user_prompt(program_name="CNTT", program_code="7480201",
            update_cycle_id="15", context_pack=pack)
        assert "M1" in prompt

    def test_prompt_approved_source_heading(self):
        from app.services.ctdt_skills.outcome_update_skill import _build_user_prompt
        pack = _make_pack()
        pack.objective_source = "laravel_approved_objectives"
        prompt, _ = _build_user_prompt(program_name="CNTT", program_code="7480201",
            update_cycle_id="15", context_pack=pack)
        assert "ĐÃ DUYỆT TỪ HỆ THỐNG CTĐT" in prompt

    def test_prompt_fallback_source_heading(self):
        from app.services.ctdt_skills.outcome_update_skill import _build_user_prompt
        pack = _make_pack()
        pack.objective_source = "rag_latest_objective_draft_fallback"
        prompt, _ = _build_user_prompt(program_name="CNTT", program_code="7480201",
            update_cycle_id="15", context_pack=pack)
        assert "CHƯA PHẢI NỘI DUNG ĐÃ DUYỆT" in prompt

    def test_prompt_legacy_source_heading(self):
        from app.services.ctdt_skills.outcome_update_skill import _build_user_prompt
        pack = _make_pack()
        pack.objective_source = "legacy_laravel_approved_objectives"
        prompt, _ = _build_user_prompt(program_name="CNTT", program_code="7480201",
            update_cycle_id="15", context_pack=pack)
        assert "ĐỊNH DẠNG LEGACY" in prompt

    def test_prompt_none_source_not_approved_heading(self):
        from app.services.ctdt_skills.outcome_update_skill import _build_user_prompt
        pack = _make_pack(with_obj=False)
        prompt, _ = _build_user_prompt(program_name="CNTT", program_code="7480201",
            update_cycle_id="15", context_pack=pack)
        assert "ĐÃ DUYỆT" not in prompt

    def test_system_prompt_no_po1(self):
        from app.services.ctdt_skills.outcome_update_skill import _SYSTEM_PROMPT
        assert "PO1" not in _SYSTEM_PROMPT
        assert "PLO1" not in _SYSTEM_PROMPT
        assert "M1" in _SYSTEM_PROMPT
        assert "C1" in _SYSTEM_PROMPT

# ══ FIX 6: Source from context ══
class TestObjectiveSource:
    @pytest.mark.asyncio
    async def test_snapshot_source(self):
        from app.api.v1.ctdt import OutcomeDraftRequest, generate_outcomes_draft
        from app.services.ctdt_outcome_update_service import OutcomeDraftResult, OutcomeSourceSummary
        snap = {"is_completed": True, "general_objective": "Test",
            "specific_objectives": [{"code": "M1", "group": "k", "text": "T"}]}
        mock_r = OutcomeDraftResult(update_cycle_id="15", program_code="7480201", program_name="CNTT",
            draft_type="outcome_update", draft_id=None, draft_saved=False,
            payload={"proposed_outcomes": [{"code": "C1", "proposed_content": "X", "outcome_type": "knowledge",
                "mapped_objectives": [], "bloom_level": "apply"}]},
            context_pack_summary={}, source_summary=OutcomeSourceSummary(
                contexts_count=0, documents_used=[], tasks_executed=["outcome_update"], latency_ms=50),
            generation_status="generated", warnings=[],
            objective_source="laravel_approved_objectives", quality_level="good",
            outcome_count=10, group_allocation=_TABLE[10],
            outcomes_flat=[{"code": "C1", "content": "X", "group": "knowledge", "bloom_level": "apply", "mapped_objectives_codes": []}],
            outcomes_structured=[{"code": "C1", "group": "knowledge", "text": "X"}],
            outcome_texts=["C1. X"])
        body = OutcomeDraftRequest(update_cycle_id="15", approved_objective_snapshot=snap)
        with patch("app.services.ctdt_outcome_update_service.generate_outcome_update_draft", return_value=mock_r):
            resp = await generate_outcomes_draft(body=body, request=AsyncMock(),
                db=AsyncMock(), user=SimpleNamespace(tenant_id="t1", id=7), query_svc=AsyncMock())
        assert resp.objective_source == "laravel_approved_objectives"

# ══ FIX 7: Quality gate ══
class TestQualityGate:
    @pytest.mark.asyncio
    async def test_good_quality(self):
        r = await _run_skill(_llm_response(10), count=10)
        assert len(r.payload.proposed_outcomes) == 10

    @pytest.mark.asyncio
    async def test_failed_quality_deficit(self):
        r = await _run_skill(_llm_response(6), count=10)
        assert r.status in ("failed", OutcomeUpdateStatus.FAILED)

    @pytest.mark.asyncio
    async def test_truncation_warning(self):
        r = await _run_skill(_llm_response(15), count=10)
        assert len(r.payload.proposed_outcomes) == 10
        assert any("15" in w and "10" in w for w in r.warnings)

    @pytest.mark.asyncio
    async def test_service_truncation_quality_warning(self):
        r = await _run_service(_llm_response(15), count=10)
        assert len(r.outcomes_structured) == 10
        assert r.quality_level == "warning"
        assert any("sinh dư" in m for m in r.quality_messages)

    @pytest.mark.asyncio
    async def test_service_all_short_outcomes_quality_failed(self):
        data = json.loads(_llm_response(10))
        for po in data["proposed_outcomes"]:
            po["proposed_content"] = "CĐR ngắn"
        r = await _run_service(json.dumps(data), count=10)
        assert r.quality_level == "failed"
        assert any("10 chuẩn đầu ra quá ngắn" in m for m in r.quality_messages)

    @pytest.mark.asyncio
    async def test_service_one_short_outcome_quality_failed(self):
        data = json.loads(_llm_response(10))
        data["proposed_outcomes"][0]["proposed_content"] = "CĐR ngắn"
        r = await _run_service(json.dumps(data), count=10)
        assert r.quality_level == "failed"
        assert any("1 chuẩn đầu ra quá ngắn" in m for m in r.quality_messages)

    @pytest.mark.asyncio
    async def test_service_plo_code_normalization_quality_warning(self):
        data = json.loads(_llm_response(10))
        for idx, po in enumerate(data["proposed_outcomes"], start=1):
            po["code"] = f"PLO{idx}"
        r = await _run_service(json.dumps(data), count=10)
        assert [po["code"] for po in r.payload["proposed_outcomes"]] == [f"C{i}" for i in range(1, 11)]
        assert r.quality_level == "warning"
        assert any("Mã CĐR" in m and "C1..Cn" in m for m in r.quality_messages)

    @pytest.mark.asyncio
    async def test_service_exact_c_codes_quality_good(self):
        r = await _run_service(_llm_response(10), count=10)
        assert r.quality_level == "good"
        assert r.quality_messages == []

    @pytest.mark.asyncio
    async def test_service_legacy_source_quality_warning(self):
        pack = _make_pack()
        pack.objective_source = "legacy_laravel_approved_objectives"
        r = await _run_service(_llm_response(10), count=10, pack=pack)
        assert r.objective_source == "legacy_laravel_approved_objectives"
        assert r.quality_level == "warning"
        assert any("legacy" in w for w in r.warnings)
        assert any("legacy" in m for m in r.quality_messages)

    @pytest.mark.asyncio
    async def test_service_missing_evidence_quality_warning(self):
        data = json.loads(_llm_response(10))
        data["proposed_outcomes"][0]["evidence_refs"] = []
        r = await _run_service(json.dumps(data), count=10)
        assert r.quality_level == "warning"
        assert any("chưa có minh chứng" in m for m in r.quality_messages)

    @pytest.mark.asyncio
    async def test_service_missing_objective_mapping_quality_warning(self):
        data = json.loads(_llm_response(10))
        data["proposed_outcomes"][0]["mapped_objectives"] = []
        r = await _run_service(json.dumps(data), count=10)
        assert r.quality_level == "warning"
        assert any("chưa liên kết rõ" in m for m in r.quality_messages)

    @pytest.mark.asyncio
    async def test_service_too_broad_quality_warning(self):
        data = json.loads(_llm_response(10))
        data["proposed_outcomes"][0]["proposed_content"] = (
            "Đào tạo nguồn nhân lực có phẩm chất tốt, phát triển toàn diện "
            "và đáp ứng nhu cầu xã hội trong lĩnh vực công nghệ thông tin."
        )
        r = await _run_service(json.dumps(data), count=10)
        assert r.quality_level == "warning"
        assert any("quá khái quát" in m for m in r.quality_messages)

    def test_response_has_quality_fields(self):
        from app.api.v1.ctdt import OutcomeDraftResponse
        fields = OutcomeDraftResponse.model_fields
        assert "quality_level" in fields
        assert "quality_messages" in fields
        assert "outcomes_structured" in fields
        assert "outcome_texts" in fields
        assert "format_profile" in fields

# ══ FIX 8: Persistence ══
class TestPersistence:
    @pytest.mark.asyncio
    async def test_save_draft_has_flat(self):
        from app.services.ctdt_outcome_update_service import generate_outcome_update_draft
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        mock_db = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()
        captured = {}
        with patch("app.services.ctdt_outcome_update_service.build_outcome_update_context_pack",
                    return_value=_make_pack()), \
             patch.object(OutcomeUpdateSkill, "_call_openai", return_value=_llm_response(10)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms, \
             patch("app.db.models.ctdt_analysis_draft.CTDTAnalysisDraft") as MockDraft:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            mock_instance = MagicMock(id=42)
            def side_effect(**kwargs):
                captured["payload"] = kwargs.get("result_payload", {})
                mock_instance.result_payload = captured["payload"]
                return mock_instance
            MockDraft.side_effect = side_effect
            result = await generate_outcome_update_draft(
                mock_db, tenant_id="t1", user_id=7, update_cycle_id="15",
                save_draft=True, outcome_count=10)
        flat = captured["payload"].get("_flat", {})
        assert flat.get("outcome_count") == 10
        assert flat.get("format_profile") == "tay_nguyen_mau_07"
        assert "outcomes_structured" in flat
        assert "outcome_texts" in flat
        assert "objective_source" in flat
        assert "quality_level" in flat
        assert "quality_messages" in flat

# ══ FIX 9: Strict allocation validation ══
    @pytest.mark.asyncio
    async def test_save_draft_persists_code_normalization_warning(self):
        from app.services.ctdt_outcome_update_service import generate_outcome_update_draft
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_llm_response(10))
        for idx, po in enumerate(data["proposed_outcomes"], start=1):
            po["code"] = f"PLO{idx}"
        mock_db = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()
        captured = {}
        with patch("app.services.ctdt_outcome_update_service.build_outcome_update_context_pack",
                    return_value=_make_pack()), \
             patch.object(OutcomeUpdateSkill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms, \
             patch("app.db.models.ctdt_analysis_draft.CTDTAnalysisDraft") as MockDraft:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            mock_instance = MagicMock(id=42)
            def side_effect(**kwargs):
                captured["payload"] = kwargs.get("result_payload", {})
                mock_instance.result_payload = captured["payload"]
                return mock_instance
            MockDraft.side_effect = side_effect
            result = await generate_outcome_update_draft(
                mock_db, tenant_id="t1", user_id=7, update_cycle_id="15",
                save_draft=True, outcome_count=10)
        flat = captured["payload"].get("_flat", {})
        assert result.quality_level == "warning"
        assert flat.get("quality_level") == "warning"
        assert any("chuẩn hóa mã" in w for w in flat.get("warnings", []))
        assert any("Mã CĐR" in m for m in flat.get("quality_messages", []))

    @pytest.mark.asyncio
    async def test_save_draft_persists_missing_evidence_warning_quality(self):
        from app.services.ctdt_outcome_update_service import generate_outcome_update_draft
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_llm_response(10))
        data["proposed_outcomes"][0]["evidence_refs"] = []
        mock_db = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()
        captured = {}
        with patch("app.services.ctdt_outcome_update_service.build_outcome_update_context_pack",
                    return_value=_make_pack()), \
             patch.object(OutcomeUpdateSkill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms, \
             patch("app.db.models.ctdt_analysis_draft.CTDTAnalysisDraft") as MockDraft:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            mock_instance = MagicMock(id=42)
            def side_effect(**kwargs):
                captured["payload"] = kwargs.get("result_payload", {})
                mock_instance.result_payload = captured["payload"]
                return mock_instance
            MockDraft.side_effect = side_effect
            result = await generate_outcome_update_draft(
                mock_db, tenant_id="t1", user_id=7, update_cycle_id="15",
                save_draft=True, outcome_count=10)
        flat = captured["payload"].get("_flat", {})
        assert result.quality_level == "warning"
        assert flat.get("quality_level") == "warning"
        assert any("minh chứng" in m for m in flat.get("quality_messages", []))

    @pytest.mark.asyncio
    async def test_save_draft_persists_short_text_failed_quality(self):
        from app.services.ctdt_outcome_update_service import generate_outcome_update_draft
        from app.services.ctdt_skills.outcome_update_skill import OutcomeUpdateSkill
        data = json.loads(_llm_response(10))
        data["proposed_outcomes"][0]["proposed_content"] = "CĐR ngắn"
        mock_db = AsyncMock()
        mock_db.flush = AsyncMock()
        mock_db.refresh = AsyncMock()
        mock_db.commit = AsyncMock()
        mock_db.add = MagicMock()
        captured = {}
        with patch("app.services.ctdt_outcome_update_service.build_outcome_update_context_pack",
                    return_value=_make_pack()), \
             patch.object(OutcomeUpdateSkill, "_call_openai", return_value=json.dumps(data)), \
             patch("app.services.ctdt_skills.outcome_update_skill.settings") as ms, \
             patch("app.db.models.ctdt_analysis_draft.CTDTAnalysisDraft") as MockDraft:
            ms.SYNTHESIS_ENABLED = True; ms.OPENAI_API_KEY = "sk-test"
            mock_instance = MagicMock(id=42)
            def side_effect(**kwargs):
                captured["payload"] = kwargs.get("result_payload", {})
                mock_instance.result_payload = captured["payload"]
                return mock_instance
            MockDraft.side_effect = side_effect
            result = await generate_outcome_update_draft(
                mock_db, tenant_id="t1", user_id=7, update_cycle_id="15",
                save_draft=True, outcome_count=10)
        flat = captured["payload"].get("_flat", {})
        assert result.quality_level == "failed"
        assert flat.get("quality_level") == "failed"
        assert any("quá ngắn" in m for m in flat.get("quality_messages", []))

class TestStrictAllocation:
    @pytest.mark.asyncio
    async def test_negative_value_422(self):
        from app.api.v1.ctdt import OutcomeDraftRequest, generate_outcomes_draft
        body = OutcomeDraftRequest(update_cycle_id="15", outcome_count=10,
            group_allocation={"knowledge": -1, "skills": 9, "autonomy_responsibility": 2})
        with pytest.raises(HTTPException) as exc_info:
            await generate_outcomes_draft(body=body, request=AsyncMock(),
                db=AsyncMock(), user=SimpleNamespace(tenant_id="t1", id=7), query_svc=AsyncMock())
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_group_422(self):
        from app.api.v1.ctdt import OutcomeDraftRequest, generate_outcomes_draft
        body = OutcomeDraftRequest(update_cycle_id="15", outcome_count=10,
            group_allocation={"knowledge": 5, "skills": 5})
        with pytest.raises(HTTPException) as exc_info:
            await generate_outcomes_draft(body=body, request=AsyncMock(),
                db=AsyncMock(), user=SimpleNamespace(tenant_id="t1", id=7), query_svc=AsyncMock())
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_wrong_distribution_422(self):
        from app.api.v1.ctdt import OutcomeDraftRequest, generate_outcomes_draft
        body = OutcomeDraftRequest(update_cycle_id="15", outcome_count=10,
            group_allocation={"knowledge": 5, "skills": 3, "autonomy_responsibility": 2})
        with pytest.raises(HTTPException) as exc_info:
            await generate_outcomes_draft(body=body, request=AsyncMock(),
                db=AsyncMock(), user=SimpleNamespace(tenant_id="t1", id=7), query_svc=AsyncMock())
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "group_allocation_contract_mismatch"

    @pytest.mark.asyncio
    async def test_exact_table_accepted(self):
        from app.api.v1.ctdt import OutcomeDraftRequest, generate_outcomes_draft
        from app.services.ctdt_outcome_update_service import OutcomeDraftResult, OutcomeSourceSummary
        mock_r = OutcomeDraftResult(update_cycle_id="15", program_code="7480201", program_name="CNTT",
            draft_type="outcome_update", draft_id=None, draft_saved=False, payload={},
            context_pack_summary={}, source_summary=OutcomeSourceSummary(
                contexts_count=0, documents_used=[], tasks_executed=["outcome_update"], latency_ms=50),
            generation_status="generated", warnings=[], outcome_count=10, group_allocation=_TABLE[10])
        body = OutcomeDraftRequest(update_cycle_id="15", outcome_count=10, group_allocation=_TABLE[10])
        with patch("app.services.ctdt_outcome_update_service.generate_outcome_update_draft", return_value=mock_r):
            resp = await generate_outcomes_draft(body=body, request=AsyncMock(),
                db=AsyncMock(), user=SimpleNamespace(tenant_id="t1", id=7), query_svc=AsyncMock())
        assert resp.outcome_count == 10

# ══ Backward compat ══
class TestBackwardCompat:
    @pytest.mark.asyncio
    async def test_old_request_defaults(self):
        from app.api.v1.ctdt import OutcomeDraftRequest, generate_outcomes_draft
        from app.services.ctdt_outcome_update_service import OutcomeDraftResult, OutcomeSourceSummary
        mock_r = OutcomeDraftResult(update_cycle_id="15", program_code="7480201", program_name="CNTT",
            draft_type="outcome_update", draft_id=None, draft_saved=False, payload={},
            context_pack_summary={}, source_summary=OutcomeSourceSummary(
                contexts_count=0, documents_used=[], tasks_executed=["outcome_update"], latency_ms=50),
            generation_status="generated", warnings=[], outcome_count=10, group_allocation=_TABLE[10])
        body = OutcomeDraftRequest(update_cycle_id="15")
        with patch("app.services.ctdt_outcome_update_service.generate_outcome_update_draft", return_value=mock_r):
            resp = await generate_outcomes_draft(body=body, request=AsyncMock(),
                db=AsyncMock(), user=SimpleNamespace(tenant_id="t1", id=7), query_svc=AsyncMock())
        assert resp.outcome_count == 10
        assert resp.draft_type == "outcome_update"

# ══ C-code sequence ══
class TestCCodeSequence:
    @pytest.mark.asyncio
    async def test_sequential(self):
        r = await _run_skill(_llm_response(10), count=10)
        codes = [po["code"] for po in r.payload.proposed_outcomes]
        assert codes == [f"C{i}" for i in range(1, 11)]

# ── Helper ──
def _empty_ret(kwargs):
    from app.services.ctdt_retrieval_service import CTDTRetrievalResult
    return CTDTRetrievalResult(query=kwargs.get("query", "q"), update_cycle_id="15", program_code="7480201",
        task_type="outcome_suggestion", document_roles_used=kwargs.get("document_roles", []),
        contexts=[], scoped_document_count=0, latency_ms=10)
