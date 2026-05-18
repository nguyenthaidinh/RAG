"""Tests for R6.3B Mapping Draft Builder V1."""
from __future__ import annotations
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException

_SVC = "app.services.ctdt_mapping_draft_builder_service"


# ── Helpers ──────────────────────────────────────────────────────────

def _make_obj_draft(*, objectives=None, draft_id=1):
    d = MagicMock()
    d.id = draft_id
    d.result_payload = {"proposed_objectives": objectives or []}
    return d

def _make_out_draft(*, outcomes=None, draft_id=2):
    d = MagicMock()
    d.id = draft_id
    d.result_payload = {"proposed_outcomes": outcomes or []}
    return d

def _obj(code="PO1", content="Đào tạo nhân lực CNTT"):
    return {"code": code, "proposed_content": content}

def _outcome(code="PLO1", content="Vận dụng AI/ML", mapped=None,
             confidence="high", evidence_refs=None):
    mapped = mapped if mapped is not None else [
        {"objective_code": "PO1", "objective_content": "CNTT", "mapping_reason": "liên quan trực tiếp"},
    ]
    return {
        "code": code, "proposed_content": content,
        "mapped_objectives": mapped,
        "confidence": confidence,
        "evidence_refs": evidence_refs or [{"ai_document_id": 10, "chunk_id": 5}],
    }


# ══ 1. Both drafts → builds objective_outcome_rows ══
class TestBuildRows:
    @pytest.mark.asyncio
    async def test_builds_rows(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        rows = r.payload["objective_outcome_rows"]
        assert len(rows) >= 1


# ══ 2. Row maps PO1→PLO1 ══
class TestRowMapping:
    @pytest.mark.asyncio
    async def test_row_codes(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        row = r.payload["objective_outcome_rows"][0]
        assert row["objective_code"] == "PO1"
        assert row["outcome_code"] == "PLO1"


# ══ 3. Default contribution_level=1 with warning ══
class TestDefaultContribution:
    @pytest.mark.asyncio
    async def test_default_contribution(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        row = r.payload["objective_outcome_rows"][0]
        assert row["contribution_level"] == 1
        assert any("defaulted to 1" in w for w in row["warnings"])


# ══ 4. source_refs from evidence_refs ══
class TestSourceRefs:
    @pytest.mark.asyncio
    async def test_source_refs_preserved(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome(
            evidence_refs=[{"ai_document_id": 42, "chunk_id": 7, "filename": "ct.pdf"}]
        )])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(
                AsyncMock(), tenant_id="t1", user_id=1,
                update_cycle_id="15", program_code="7480201", program_id="p1",
            )
        row = r.payload["objective_outcome_rows"][0]
        assert len(row["source_refs"]) == 1
        ref = row["source_refs"][0]
        assert ref["ai_document_id"] == 42
        assert ref["update_cycle_id"] == "15"
        assert ref["program_code"] == "7480201"
        assert ref["program_id"] == "p1"


# ══ 5. evidence_ref without ai_document_id skipped ══
class TestNoFakeSource:
    @pytest.mark.asyncio
    async def test_no_fake_source(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome(
            evidence_refs=[{"source_index": 0}]  # no ai_document_id
        )])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        row = r.payload["objective_outcome_rows"][0]
        assert row["source_refs"] == []


# ══ 6. Invalid confidence normalized ══
class TestConfidenceNormalize:
    @pytest.mark.asyncio
    async def test_invalid_confidence(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome(confidence="very_high")])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        row = r.payload["objective_outcome_rows"][0]
        assert row["confidence"] == "medium"


# ══ 7. Mapped objective not found → needs_review ══
class TestObjectiveNotFound:
    @pytest.mark.asyncio
    async def test_not_found(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj("PO1")])
        out_d = _make_out_draft(outcomes=[_outcome(
            mapped=[{"objective_code": "PO99", "objective_content": "", "mapping_reason": ""}]
        )])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        row = r.payload["objective_outcome_rows"][0]
        assert row["status"] == "needs_review"
        assert row["confidence"] == "low"
        assert "mapped_objective_not_found" in row["warnings"]


# ══ 8. Missing objective_update draft ══
class TestMissingObjDraft:
    @pytest.mark.asyncio
    async def test_missing_obj(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[None, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        mi_types = [m["type"] for m in r.missing_information]
        assert "objective_update" in mi_types
        assert r.payload["objective_outcome_rows"] == []


# ══ 9. Missing outcome_update draft ══
class TestMissingOutDraft:
    @pytest.mark.asyncio
    async def test_missing_out(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, None]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        mi_types = [m["type"] for m in r.missing_information]
        assert "outcome_update" in mi_types
        assert r.payload["objective_outcome_rows"] == []


# ══ 10. V1 no course_outcome rows ══
class TestNoCourseRows:
    @pytest.mark.asyncio
    async def test_empty_course_rows(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        assert r.payload["course_outcome_rows"] == []


# ══ 11. V1 no CLO rows ══
class TestNoCLORows:
    @pytest.mark.asyncio
    async def test_empty_clo_rows(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        assert r.payload["clo_program_outcome_rows"] == []


# ══ 12. source_summary canonical keys ══
class TestSourceSummaryKeys:
    @pytest.mark.asyncio
    async def test_canonical_keys(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        rc = r.source_summary["rows_count"]
        assert set(rc.keys()) == {
            "objective_outcome", "course_outcome",
            "course_learning_outcome_program_outcome",
        }


# ══ 13. save_draft=false no commit ══
class TestNoCommit:
    @pytest.mark.asyncio
    async def test_no_commit(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        db = AsyncMock()
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(db, tenant_id="t1", user_id=1, update_cycle_id="15", save_draft=False)
        db.commit.assert_not_called()
        assert r.draft_saved is False
        assert r.draft_id is None


# ══ 14. save_draft=true persists ══
class TestSaveDraft:
    @pytest.mark.asyncio
    async def test_save_draft(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        mock_draft = MagicMock()
        mock_draft.id = 42
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]), \
             patch(f"{_SVC}.save_raw_analysis_draft", return_value=mock_draft) as save_mock:
            db = AsyncMock()
            r = await build_mapping_draft(db, tenant_id="t1", user_id=1, update_cycle_id="15", save_draft=True)
        assert r.draft_saved is True
        assert r.draft_id == 42
        save_mock.assert_called_once()
        call_kwargs = save_mock.call_args[1]
        assert call_kwargs["draft_type"] == "mapping_draft"
        assert call_kwargs["analysis_mode"] == "design"
        db.commit.assert_called_once()


# ══ 15. Endpoint returns draft_type="mapping_draft" ══
class TestEndpointResponse:
    @pytest.mark.asyncio
    async def test_endpoint_draft_type(self):
        from app.api.v1.ctdt import build_mapping_draft_endpoint
        from app.services.ctdt_mapping_draft_builder_service import MappingDraftBuildResult
        mock_result = MappingDraftBuildResult(
            update_cycle_id="15",
            payload={"draft_type": "mapping_draft", "objective_outcome_rows": [],
                     "course_outcome_rows": [], "clo_program_outcome_rows": [],
                     "warnings": [], "source_summary": {}},
            source_summary={"documents_used": [], "rows_count": {}, "source_types": []},
            missing_information=[], warnings=[],
        )
        with patch(f"{_SVC}.build_mapping_draft", return_value=mock_result):
            resp = await build_mapping_draft_endpoint(
                body=SimpleNamespace(
                    update_cycle_id="15", program_id=None, program_code=None,
                    program_name=None, mapping_types=["objective_outcome"],
                    save_draft=False,
                ),
                request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )
        assert resp.draft_type == "mapping_draft"


# ══ 16. No LLM imports in builder ══
class TestNoLLMImports:
    def test_no_llm(self):
        import app.services.ctdt_mapping_draft_builder_service as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "openai" not in joined.lower()
        assert "httpx" not in joined.lower()


# ══ 17. No Program model imports ══
class TestNoProgramImports:
    def test_no_program(self):
        import app.services.ctdt_mapping_draft_builder_service as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "ProgramVersion" not in joined
        assert "from app.db.models.program" not in joined


# ══ 18. Endpoint error handling ══
class TestEndpointError:
    @pytest.mark.asyncio
    async def test_error_500(self):
        from app.api.v1.ctdt import build_mapping_draft_endpoint
        with patch(f"{_SVC}.build_mapping_draft", side_effect=RuntimeError("boom")):
            with pytest.raises(HTTPException) as exc_info:
                await build_mapping_draft_endpoint(
                    body=SimpleNamespace(
                        update_cycle_id="15", program_id=None, program_code=None,
                        program_name=None, mapping_types=["objective_outcome"],
                        save_draft=False,
                    ),
                    request=AsyncMock(), db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                )
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "mapping_draft_build_error"


# ══ 19. V1 warning present ══
class TestV1Warning:
    @pytest.mark.asyncio
    async def test_v1_warning(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        assert any("V1" in w for w in r.warnings)


# ══ 20. get_latest_analysis_draft receives program_id ══
class TestProgramIdScoping:
    @pytest.mark.asyncio
    async def test_program_id_passed(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]) as mock_get:
            await build_mapping_draft(
                AsyncMock(), tenant_id="t1", user_id=1,
                update_cycle_id="15", program_id="p1",
            )
        for call in mock_get.call_args_list:
            assert call[1]["program_id"] == "p1"


# ══ 21. Unsupported mapping_type ignored ══
class TestUnsupportedMappingType:
    @pytest.mark.asyncio
    async def test_unsupported_ignored(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(
                AsyncMock(), tenant_id="t1", user_id=1,
                update_cycle_id="15", mapping_types=["invalid_type"],
            )
        assert r.payload["objective_outcome_rows"] == []
        assert any("Unsupported" in w for w in r.warnings)


# ══ 22. Deferred mapping_type warning ══
class TestDeferredMappingType:
    @pytest.mark.asyncio
    async def test_deferred_warning(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(
                AsyncMock(), tenant_id="t1", user_id=1,
                update_cycle_id="15", mapping_types=["course_outcome"],
            )
        assert r.payload["course_outcome_rows"] == []
        assert any("deferred" in w for w in r.warnings)


# ══ 23. Empty mapping_types defaults ══
class TestEmptyMappingTypes:
    @pytest.mark.asyncio
    async def test_empty_defaults(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(
                AsyncMock(), tenant_id="t1", user_id=1,
                update_cycle_id="15", mapping_types=[],
            )
        # Empty list → defaults to objective_outcome
        assert len(r.payload["objective_outcome_rows"]) >= 1


# ══ 24. save_draft=true + missing deps skips save ══
class TestSaveSkipMissingDeps:
    @pytest.mark.asyncio
    async def test_skip_save_missing(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, None]), \
             patch(f"{_SVC}.save_raw_analysis_draft") as save_mock:
            r = await build_mapping_draft(
                AsyncMock(), tenant_id="t1", user_id=1,
                update_cycle_id="15", save_draft=True,
            )
        save_mock.assert_not_called()
        assert r.draft_saved is False
        assert any("not saved" in w for w in r.warnings)


# ══ 25. save_draft=true + save error raises MappingDraftSaveError ══
class TestSaveErrorRaises:
    @pytest.mark.asyncio
    async def test_save_error(self):
        from app.services.ctdt_mapping_draft_builder_service import (
            build_mapping_draft, MappingDraftSaveError,
        )
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        db = AsyncMock()
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]), \
             patch(f"{_SVC}.save_raw_analysis_draft", side_effect=RuntimeError("db boom")):
            with pytest.raises(MappingDraftSaveError):
                await build_mapping_draft(
                    db, tenant_id="t1", user_id=1,
                    update_cycle_id="15", save_draft=True,
                )
        db.rollback.assert_called_once()


# ══ 26. Endpoint catches MappingDraftSaveError ══
class TestEndpointSaveError:
    @pytest.mark.asyncio
    async def test_save_error_500(self):
        from app.api.v1.ctdt import build_mapping_draft_endpoint
        from app.services.ctdt_mapping_draft_builder_service import MappingDraftSaveError
        with patch(f"{_SVC}.build_mapping_draft", side_effect=MappingDraftSaveError("fail")):
            with pytest.raises(HTTPException) as exc_info:
                await build_mapping_draft_endpoint(
                    body=SimpleNamespace(
                        update_cycle_id="15", program_id=None, program_code=None,
                        program_name=None, mapping_types=["objective_outcome"],
                        save_draft=True,
                    ),
                    request=AsyncMock(), db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                )
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "mapping_draft_save_error"


# ══ 27. Duplicate mapped_objectives → one row ══
class TestDedupRows:
    @pytest.mark.asyncio
    async def test_dedupe(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome(
            mapped=[
                {"objective_code": "PO1", "objective_content": "", "mapping_reason": "r1"},
                {"objective_code": "PO1", "objective_content": "", "mapping_reason": "r2"},
            ]
        )])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]):
            r = await build_mapping_draft(AsyncMock(), tenant_id="t1", user_id=1, update_cycle_id="15")
        rows = r.payload["objective_outcome_rows"]
        assert len(rows) == 1
        assert any("Duplicate" in w for w in r.warnings)


# ══ 28. MappingDraftBuildRequest default_factory isolation ══
class TestRequestDefaultFactory:
    def test_isolation(self):
        from app.api.v1.ctdt import MappingDraftBuildRequest
        r1 = MappingDraftBuildRequest(update_cycle_id="a")
        r2 = MappingDraftBuildRequest(update_cycle_id="b")
        r1.mapping_types.append("extra")
        assert "extra" not in r2.mapping_types


# ══ 29. save_draft=true + invalid mapping_type → no save ══
class TestSaveGuardInvalidType:
    @pytest.mark.asyncio
    async def test_no_save_invalid_type(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]), \
             patch(f"{_SVC}.save_raw_analysis_draft") as save_mock:
            r = await build_mapping_draft(
                AsyncMock(), tenant_id="t1", user_id=1,
                update_cycle_id="15", save_draft=True,
                mapping_types=["invalid_type"],
            )
        save_mock.assert_not_called()
        assert r.draft_saved is False
        assert any("no supported mapping type" in w for w in r.warnings)


# ══ 30. save_draft=true + deferred mapping_type → no save ══
class TestSaveGuardDeferredType:
    @pytest.mark.asyncio
    async def test_no_save_deferred_type(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]), \
             patch(f"{_SVC}.save_raw_analysis_draft") as save_mock:
            r = await build_mapping_draft(
                AsyncMock(), tenant_id="t1", user_id=1,
                update_cycle_id="15", save_draft=True,
                mapping_types=["course_outcome"],
            )
        save_mock.assert_not_called()
        assert r.draft_saved is False
        assert any("no supported mapping type" in w for w in r.warnings)


# ══ 31. save_draft=true + mapping_types=[] → default saves OK ══
class TestSaveGuardEmptyDefault:
    @pytest.mark.asyncio
    async def test_empty_defaults_and_saves(self):
        from app.services.ctdt_mapping_draft_builder_service import build_mapping_draft
        obj_d = _make_obj_draft(objectives=[_obj()])
        out_d = _make_out_draft(outcomes=[_outcome()])
        mock_draft = MagicMock()
        mock_draft.id = 99
        with patch(f"{_SVC}.get_latest_analysis_draft", side_effect=[obj_d, out_d]), \
             patch(f"{_SVC}.save_raw_analysis_draft", return_value=mock_draft) as save_mock:
            db = AsyncMock()
            r = await build_mapping_draft(
                db, tenant_id="t1", user_id=1,
                update_cycle_id="15", save_draft=True,
                mapping_types=[],
            )
        save_mock.assert_called_once()
        assert r.draft_saved is True
        assert r.draft_id == 99
