"""Tests for R6.3C Mapping Draft Latest Endpoint."""
from __future__ import annotations
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import pytest
from fastapi import HTTPException

_DRAFT_SVC = "app.services.ctdt_analysis_draft_service"


# ── Helpers ──────────────────────────────────────────────────────────

def _make_draft(*, draft_id=12, payload=None, source_summary=None):
    d = MagicMock()
    d.id = draft_id
    d.update_cycle_id = "15"
    d.program_id = "p1"
    d.program_code = "7480201"
    d.program_name = "CNTT"
    d.draft_type = "mapping_draft"
    d.analysis_mode = "design"
    d.status = "draft"
    d.result_payload = payload or {
        "draft_type": "mapping_draft",
        "objective_outcome_rows": [
            {"objective_code": "PO1", "outcome_code": "PLO1", "contribution_level": 1},
        ],
        "course_outcome_rows": [],
        "clo_program_outcome_rows": [],
        "warnings": [],
        "source_summary": {
            "documents_used": [],
            "rows_count": {
                "objective_outcome": 1,
                "course_outcome": 0,
                "course_learning_outcome_program_outcome": 0,
            },
            "source_types": ["generated_from_draft"],
        },
    }
    d.source_summary = source_summary or {
        "documents_used": [],
        "rows_count": {
            "objective_outcome": 1,
            "course_outcome": 0,
            "course_learning_outcome_program_outcome": 0,
        },
        "source_types": ["generated_from_draft"],
    }
    d.created_at = datetime(2026, 5, 17, 10, 0, 0)
    d.updated_at = datetime(2026, 5, 17, 10, 0, 0)
    return d


# ══ 1. Endpoint returns draft_type="mapping_draft" ══
class TestDraftType:
    @pytest.mark.asyncio
    async def test_draft_type(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        draft = _make_draft()
        with patch(f"{_DRAFT_SVC}.get_latest_analysis_draft", return_value=draft):
            resp = await get_latest_mapping_draft(
                update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )
        assert resp.draft_type == "mapping_draft"


# ══ 2. Endpoint calls get_latest_analysis_draft with draft_type="mapping_draft" ══
class TestCallsDraftService:
    @pytest.mark.asyncio
    async def test_calls_with_mapping_draft(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        draft = _make_draft()
        with patch(f"{_DRAFT_SVC}.get_latest_analysis_draft", return_value=draft) as mock_get:
            await get_latest_mapping_draft(
                update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )
        call_kwargs = mock_get.call_args[1]
        assert call_kwargs["draft_type"] == "mapping_draft"


# ══ 3. Endpoint passes program_id ══
class TestProgramIdPassed:
    @pytest.mark.asyncio
    async def test_program_id(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        draft = _make_draft()
        with patch(f"{_DRAFT_SVC}.get_latest_analysis_draft", return_value=draft) as mock_get:
            await get_latest_mapping_draft(
                update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                program_id="p1",
            )
        assert mock_get.call_args[1]["program_id"] == "p1"


# ══ 4. Endpoint passes program_code ══
class TestProgramCodePassed:
    @pytest.mark.asyncio
    async def test_program_code(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        draft = _make_draft()
        with patch(f"{_DRAFT_SVC}.get_latest_analysis_draft", return_value=draft) as mock_get:
            await get_latest_mapping_draft(
                update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                program_code="7480201",
            )
        assert mock_get.call_args[1]["program_code"] == "7480201"


# ══ 5. Draft not found → 404 ══
class TestNotFound:
    @pytest.mark.asyncio
    async def test_404(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        with patch(f"{_DRAFT_SVC}.get_latest_analysis_draft", return_value=None):
            with pytest.raises(HTTPException) as exc_info:
                await get_latest_mapping_draft(
                    update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                )
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["code"] == "mapping_draft_not_found"


# ══ 6. analysis_mode != design → 422 ══
class TestInvalidMode:
    @pytest.mark.asyncio
    async def test_422(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        with pytest.raises(HTTPException) as exc_info:
            await get_latest_mapping_draft(
                update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                analysis_mode="review",
            )
        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "mapping_draft_requires_design_mode"


# ══ 7. Response contains objective_outcome_rows ══
class TestPayloadRows:
    @pytest.mark.asyncio
    async def test_rows_preserved(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        draft = _make_draft()
        with patch(f"{_DRAFT_SVC}.get_latest_analysis_draft", return_value=draft):
            resp = await get_latest_mapping_draft(
                update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )
        rows = resp.payload["objective_outcome_rows"]
        assert len(rows) == 1
        assert rows[0]["objective_code"] == "PO1"


# ══ 8. Response rows_count canonical keys ══
class TestRowsCountKeys:
    @pytest.mark.asyncio
    async def test_canonical_keys(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        draft = _make_draft()
        with patch(f"{_DRAFT_SVC}.get_latest_analysis_draft", return_value=draft):
            resp = await get_latest_mapping_draft(
                update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )
        rc = resp.source_summary["rows_count"]
        assert set(rc.keys()) == {
            "objective_outcome", "course_outcome",
            "course_learning_outcome_program_outcome",
        }


# ══ 9. No LLM imports in endpoint module ══
class TestNoLLMImports:
    def test_no_llm(self):
        """Endpoint itself doesn't import LLM; only uses draft service."""
        import app.services.ctdt_analysis_draft_service as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "openai" not in joined.lower()
        assert "httpx" not in joined.lower()


# ══ 10. Endpoint does not commit DB ══
class TestNoCommit:
    @pytest.mark.asyncio
    async def test_no_commit(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        draft = _make_draft()
        db = AsyncMock()
        with patch(f"{_DRAFT_SVC}.get_latest_analysis_draft", return_value=draft):
            await get_latest_mapping_draft(
                update_cycle_id="15", request=AsyncMock(), db=db,
                user=SimpleNamespace(tenant_id="t1", id=7),
            )
        db.commit.assert_not_called()


# ══ 11. No Program model imports ══
class TestNoProgramImports:
    def test_no_program(self):
        import app.services.ctdt_analysis_draft_service as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "ProgramVersion" not in joined
        assert "from app.db.models.program" not in joined


# ══ 12. Server error → 500 ══
class TestServerError:
    @pytest.mark.asyncio
    async def test_500(self):
        from app.api.v1.ctdt import get_latest_mapping_draft
        with patch(f"{_DRAFT_SVC}.get_latest_analysis_draft", side_effect=RuntimeError("db")):
            with pytest.raises(HTTPException) as exc_info:
                await get_latest_mapping_draft(
                    update_cycle_id="15", request=AsyncMock(), db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                )
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "mapping_draft_latest_error"
