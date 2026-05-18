"""Tests for R6.2C Objective ↔ Outcome Alignment Review."""
from __future__ import annotations
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException


# ── Helpers ──────────────────────────────────────────────────────────

def _make_obj_draft(*, objectives=None, draft_id=1):
    d = MagicMock()
    d.id = draft_id
    d.result_payload = {"proposed_objectives": objectives or []}
    return d

def _make_out_draft(*, outcomes=None, alignment=None, draft_id=2):
    d = MagicMock()
    d.id = draft_id
    payload = {"proposed_outcomes": outcomes or []}
    if alignment is not None:
        payload["objective_outcome_alignment"] = alignment
    d.result_payload = payload
    return d

def _obj(code="PO1", content="Đào tạo nhân lực CNTT"):
    return {"code": code, "proposed_content": content}

def _outcome(code="PLO1", content="Vận dụng AI/ML", mapped=None, confidence="high",
             quality_flags=None, evidence_refs=None):
    mapped = mapped if mapped is not None else [{"objective_code": "PO1", "objective_content": "CNTT", "mapping_reason": "ok"}]
    return {
        "code": code, "proposed_content": content, "mapped_objectives": mapped,
        "confidence": confidence, "quality_flags": quality_flags or [],
        "evidence_refs": evidence_refs or [{"source_index": 0}],
    }


# ══ 1. Covered objective + valid outcome ══
class TestCoveredValid:
    @pytest.mark.asyncio
    async def test_covered_objective_valid_outcome(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        cov = r["objective_coverage"]
        assert len(cov) == 1
        assert cov[0]["coverage_status"] == "covered"
        qual = r["outcome_mapping_quality"]
        assert qual[0]["mapping_status"] == "valid"
        assert r["summary"]["covered_objectives_count"] == 1


# ══ 2. Objective not covered ══
class TestObjectiveNotCovered:
    @pytest.mark.asyncio
    async def test_objective_not_covered_gap(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj("PO1"), _obj("PO2", "Nghiên cứu")])
        out_d = _make_out_draft(outcomes=[_outcome("PLO1", mapped=[{"objective_code": "PO1", "objective_content": "", "mapping_reason": ""}])])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        not_cov = [c for c in r["objective_coverage"] if c["coverage_status"] == "not_covered"]
        assert len(not_cov) == 1
        assert not_cov[0]["objective_code"] == "PO2"
        gaps = [g for g in r["gaps"] if g["type"] == "objective_not_covered"]
        assert len(gaps) == 1
        assert gaps[0]["severity"] == "high"


# ══ 3. Outcome unmapped ══
class TestOutcomeUnmapped:
    @pytest.mark.asyncio
    async def test_outcome_unmapped_gap(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome("PLO1", mapped=[])])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        qual = r["outcome_mapping_quality"]
        assert qual[0]["mapping_status"] == "missing"
        gaps = [g for g in r["gaps"] if g["type"] == "outcome_unmapped"]
        assert len(gaps) == 1
        assert gaps[0]["severity"] == "high"


# ══ 4. Mapped objective not found → weak ══
class TestMappedObjectiveNotFound:
    @pytest.mark.asyncio
    async def test_mapped_objective_not_found(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj("PO1")])
        out_d = _make_out_draft(outcomes=[_outcome("PLO1", mapped=[{"objective_code": "PO99", "objective_content": "", "mapping_reason": ""}])])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        assert r["outcome_mapping_quality"][0]["mapping_status"] == "weak"
        assert "mapped_objective_not_found" in r["outcome_mapping_quality"][0]["issues"]


# ══ 5. Low confidence gap ══
class TestLowConfidence:
    @pytest.mark.asyncio
    async def test_low_confidence_gap(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome("PLO1", confidence="low")])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        gaps = [g for g in r["gaps"] if g["type"] == "low_confidence"]
        assert len(gaps) == 1
        assert gaps[0]["severity"] == "medium"


# ══ 6. Missing evidence gap ══
class TestMissingEvidence:
    @pytest.mark.asyncio
    async def test_missing_evidence_gap(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome("PLO1", quality_flags=["missing_evidence"])])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        gaps = [g for g in r["gaps"] if g["type"] == "missing_evidence"]
        assert len(gaps) == 1
        assert gaps[0]["severity"] == "medium"


# ══ 7. Needs human review gap ══
class TestNeedsHumanReview:
    @pytest.mark.asyncio
    async def test_needs_human_review_gap(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome("PLO1", quality_flags=["needs_human_review"])])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        gaps = [g for g in r["gaps"] if g["type"] == "needs_human_review"]
        assert len(gaps) == 1


# ══ 8. Missing objective_update draft ══
class TestMissingObjectiveDraft:
    @pytest.mark.asyncio
    async def test_missing_objective_draft(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[None, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        mi_types = [m["type"] for m in r["missing_information"]]
        assert "objective_update" in mi_types
        assert r["objective_draft_id"] is None
        # R6.2C.1: no false gaps when objective draft missing
        assert r["gaps"] == []
        assert r["objective_coverage"] == []
        assert r["outcome_mapping_quality"] == []
        assert r["summary"]["objectives_count"] == 0
        assert r["summary"]["outcomes_count"] == 1


# ══ 9. Missing outcome_update draft ══
class TestMissingOutcomeDraft:
    @pytest.mark.asyncio
    async def test_missing_outcome_draft(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj()])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, None]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        mi_types = [m["type"] for m in r["missing_information"]]
        assert "outcome_update" in mi_types
        assert r["outcome_draft_id"] is None
        # R6.2C.1: no false gaps when outcome draft missing
        assert r["gaps"] == []
        assert r["objective_coverage"] == []
        assert r["outcome_mapping_quality"] == []
        assert r["summary"]["objectives_count"] == 1
        assert r["summary"]["outcomes_count"] == 0


# ══ 10. Both drafts missing ══
class TestBothDraftsMissing:
    @pytest.mark.asyncio
    async def test_both_missing(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", return_value=None):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        assert r["summary"]["objectives_count"] == 0
        assert r["summary"]["outcomes_count"] == 0
        assert len(r["gaps"]) == 0
        assert len(r["missing_information"]) == 2
        assert len(r["next_actions"]) >= 2


# ══ 11. Endpoint read-only no commit ══
class TestEndpointReadOnly:
    @pytest.mark.asyncio
    async def test_endpoint_no_commit(self):
        from app.api.v1.ctdt import review_alignment_objectives_outcomes
        mock_result = {
            "update_cycle_id": "15", "program_code": None,
            "review_type": "objective_outcome_alignment",
            "objective_draft_id": 1, "outcome_draft_id": 2,
            "summary": {"objectives_count": 0, "outcomes_count": 0,
                "covered_objectives_count": 0, "partially_covered_objectives_count": 0,
                "not_covered_objectives_count": 0, "unmapped_outcomes_count": 0,
                "low_confidence_outcomes_count": 0, "needs_human_review_count": 0},
            "objective_coverage": [], "outcome_mapping_quality": [],
            "gaps": [], "missing_information": [], "next_actions": [],
        }
        db = AsyncMock()
        with patch("app.services.ctdt_alignment_review_service.review_objective_outcome_alignment", return_value=mock_result):
            resp = await review_alignment_objectives_outcomes(
                update_cycle_id="15", request=AsyncMock(), db=db,
                user=SimpleNamespace(tenant_id="t1", id=7))
        db.commit.assert_not_called()
        db.add.assert_not_called()
        assert resp.review_type == "objective_outcome_alignment"


# ══ 12. No LLM calls ══
class TestNoLLMCalls:
    def test_service_no_llm_imports(self):
        import app.services.ctdt_alignment_review_service as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "openai" not in joined.lower()
        assert "httpx" not in joined.lower()
        assert "SYNTHESIS" not in src.split("def ")[0]  # no synthesis in module-level


# ══ 13. No Program writes ══
class TestNoProgramWrites:
    def test_no_program_imports(self):
        import app.services.ctdt_alignment_review_service as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "ProgramVersion" not in joined
        assert "from app.db.models.program" not in joined


# ══ 14. Endpoint error handling ══
class TestEndpointError:
    @pytest.mark.asyncio
    async def test_service_error_500(self):
        from app.api.v1.ctdt import review_alignment_objectives_outcomes
        with patch("app.services.ctdt_alignment_review_service.review_objective_outcome_alignment",
                   side_effect=RuntimeError("boom")):
            with pytest.raises(HTTPException) as exc_info:
                await review_alignment_objectives_outcomes(
                    update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7))
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "alignment_review_error"


# ══ 15. Partially covered objective ══
class TestPartiallyCovered:
    @pytest.mark.asyncio
    async def test_partially_covered(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome("PLO1", confidence="low")])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        assert r["objective_coverage"][0]["coverage_status"] == "partially_covered"
        assert r["summary"]["partially_covered_objectives_count"] == 1


# ══ 16. Summary counts correct ══
class TestSummaryCounts:
    @pytest.mark.asyncio
    async def test_summary_counts(self):
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        objs = [_obj("PO1"), _obj("PO2"), _obj("PO3")]
        outs = [
            _outcome("PLO1", mapped=[{"objective_code": "PO1", "objective_content": "", "mapping_reason": ""}]),
            _outcome("PLO2", mapped=[{"objective_code": "PO2", "objective_content": "", "mapping_reason": ""}], confidence="low"),
            _outcome("PLO3", mapped=[]),
        ]
        obj_d = _make_obj_draft(objectives=objs)
        out_d = _make_out_draft(outcomes=outs)
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        s = r["summary"]
        assert s["objectives_count"] == 3
        assert s["outcomes_count"] == 3
        assert s["covered_objectives_count"] == 1
        assert s["partially_covered_objectives_count"] == 1
        assert s["not_covered_objectives_count"] == 1
        assert s["unmapped_outcomes_count"] == 1
        assert s["low_confidence_outcomes_count"] == 1


# ══ 17. No objective_not_covered gap when outcome draft missing ══
class TestNoFalseGapMissingOutcome:
    @pytest.mark.asyncio
    async def test_no_objective_not_covered_gap(self):
        """With outcome draft missing, objectives should NOT appear as not_covered."""
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        obj_d = _make_obj_draft(objectives=[_obj("PO1"), _obj("PO2")])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[obj_d, None]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        assert not any(g["type"] == "objective_not_covered" for g in r["gaps"])
        assert r["objective_coverage"] == []
        assert r["summary"]["not_covered_objectives_count"] == 0


# ══ 18. No mapped_objective_not_found gap when objective draft missing ══
class TestNoFalseGapMissingObjective:
    @pytest.mark.asyncio
    async def test_no_mapped_objective_not_found_gap(self):
        """With objective draft missing, outcomes should NOT get weak/unmapped."""
        from app.services.ctdt_alignment_review_service import review_objective_outcome_alignment
        out_d = _make_out_draft(outcomes=[
            _outcome("PLO1", mapped=[{"objective_code": "PO1", "objective_content": "", "mapping_reason": ""}]),
            _outcome("PLO2", mapped=[]),
        ])
        with patch("app.services.ctdt_alignment_review_service.get_latest_analysis_draft", side_effect=[None, out_d]):
            r = await review_objective_outcome_alignment(AsyncMock(), tenant_id="t1", update_cycle_id="15")
        assert not any(g["type"] == "outcome_unmapped" for g in r["gaps"])
        assert not any(g["type"] == "objective_not_covered" for g in r["gaps"])
        assert r["outcome_mapping_quality"] == []
