"""Tests for R5.6 Mẫu 06 dedicated draft API endpoint."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.db.models.ctdt_analysis_draft import CTDTAnalysisDraft


# ── Fixture helpers ──────────────────────────────────────────────────


def _make_draft(
    *,
    change_proposals: list | None = None,
    draft_id: int = 12,
    update_cycle_id: str = "15",
    program_code: str = "7480201",
    program_name: str = "CNTT",
    analysis_mode: str = "draft",
    status: str = "draft",
) -> CTDTAnalysisDraft:
    """Build a CTDTAnalysisDraft with the given change_proposals in result_payload."""
    if change_proposals is None:
        change_proposals = [
            {
                "status": "generated",
                "task_type": "change_proposal",
                "sources": [
                    {
                        "ai_document_id": 42,
                        "external_file_id": "ext-001",
                        "filename": "survey.pdf",
                        "document_role": "survey_evidence",
                        "chunk_id": 123,
                        "chunk_index": 0,
                        "score": 0.82,
                        "quote": "Kết quả khảo sát cho thấy...",
                        "update_cycle_id": "15",
                        "program_code": "7480201",
                    }
                ],
                "payload": {
                    "target_area": "Chuẩn đầu ra",
                    "change_type": "update",
                    "current_issue": "CĐR chưa phản ánh năng lực AI",
                    "proposed_change": "Bổ sung CĐR về AI",
                    "rationale": "Có minh chứng từ tài liệu cập nhật",
                    "expected_impact": "Sinh viên đáp ứng tốt hơn yêu cầu việc làm",
                    "priority": "high",
                    "confidence": "high",
                },
            }
        ]

    draft = CTDTAnalysisDraft(
        tenant_id="t1",
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        program_name=program_name,
        analysis_mode=analysis_mode,
        draft_type="update_cycle_analysis",
        result_payload={"change_proposals": change_proposals},
        source_summary={
            "contexts_count": 1,
            "documents_used": [42],
            "tasks_executed": ["change_proposal"],
            "latency_ms": 25,
        },
        status=status,
    )
    draft.id = draft_id
    draft.created_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    draft.updated_at = datetime.now(timezone.utc)
    return draft


# ══════════════════════════════════════════════════════════════════════
# 1. Helper extraction tests
# ══════════════════════════════════════════════════════════════════════


class TestExtractMau06Items:
    """Test _extract_mau06_items_from_draft helper."""

    def test_valid_payload_returns_items_with_proposed_change(self):
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        draft = _make_draft()
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert len(items) == 1
        assert items[0].proposed_change == "Bổ sung CĐR về AI"
        assert items[0].target_area == "Chuẩn đầu ra"
        assert items[0].change_type == "update"
        assert items[0].priority == "high"
        assert items[0].confidence == "high"
        assert warnings == []

    def test_sources_preserved_in_items(self):
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        draft = _make_draft()
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert len(items[0].sources) == 1
        src = items[0].sources[0]
        assert src.ai_document_id == 42
        assert src.external_file_id == "ext-001"
        assert src.filename == "survey.pdf"
        assert src.document_role == "survey_evidence"
        assert src.chunk_id == 123
        assert src.chunk_index == 0
        assert src.score == 0.82
        assert src.quote == "Kết quả khảo sát cho thấy..."
        assert src.update_cycle_id == "15"
        assert src.program_code == "7480201"

    def test_missing_payload_skipped_with_warning(self):
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        proposals = [
            {
                "status": "generated",
                "task_type": "change_proposal",
                "sources": [],
                # No payload key
            },
            {
                "status": "generated",
                "task_type": "change_proposal",
                "sources": [],
                "payload": None,  # Explicit None
            },
        ]
        draft = _make_draft(change_proposals=proposals)
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert len(items) == 0
        assert len(warnings) == 2
        assert "missing payload" in warnings[0].lower()
        assert "missing payload" in warnings[1].lower()

    def test_non_dict_item_skipped_with_warning(self):
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        draft = _make_draft(change_proposals=["not_a_dict", 42])
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert len(items) == 0
        assert len(warnings) == 2
        assert "not a dict" in warnings[0].lower()

    def test_empty_change_proposals_returns_empty_items(self):
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        draft = _make_draft(change_proposals=[])
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert items == []
        assert warnings == []

    def test_no_change_proposals_key_returns_empty_items(self):
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        draft = _make_draft()
        draft.result_payload = {}  # No change_proposals key
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert items == []
        assert warnings == []

    def test_change_proposals_not_list_returns_warning(self):
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        draft = _make_draft()
        draft.result_payload = {"change_proposals": "invalid"}
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert items == []
        assert len(warnings) == 1
        assert "not a list" in warnings[0].lower()

    def test_missing_payload_field_defaults_to_empty_string(self):
        """If a Mẫu 06 field is missing from payload, it defaults to ''."""
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        proposals = [
            {
                "status": "generated",
                "task_type": "change_proposal",
                "sources": [],
                "payload": {
                    "target_area": "Chuẩn đầu ra",
                    "change_type": "update",
                    # Remaining 6 fields missing
                },
            },
        ]
        draft = _make_draft(change_proposals=proposals)
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert len(items) == 1
        assert items[0].target_area == "Chuẩn đầu ra"
        assert items[0].proposed_change == ""
        assert items[0].rationale == ""
        assert warnings == []

    def test_multiple_valid_items(self):
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        proposals = [
            {
                "status": "generated",
                "task_type": "change_proposal",
                "sources": [],
                "payload": {
                    "target_area": "Chuẩn đầu ra",
                    "change_type": "update",
                    "current_issue": "Issue 1",
                    "proposed_change": "Change 1",
                    "rationale": "R1",
                    "expected_impact": "I1",
                    "priority": "high",
                    "confidence": "high",
                },
            },
            {
                "status": "generated",
                "task_type": "change_proposal",
                "sources": [],
                "payload": {
                    "target_area": "Nội dung chương trình",
                    "change_type": "add",
                    "current_issue": "Issue 2",
                    "proposed_change": "Change 2",
                    "rationale": "R2",
                    "expected_impact": "I2",
                    "priority": "medium",
                    "confidence": "medium",
                },
            },
        ]
        draft = _make_draft(change_proposals=proposals)
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert len(items) == 2
        assert items[0].proposed_change == "Change 1"
        assert items[1].proposed_change == "Change 2"
        assert items[1].target_area == "Nội dung chương trình"
        assert warnings == []

    def test_null_result_payload_returns_empty(self):
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        draft = _make_draft()
        draft.result_payload = None
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert items == []
        assert warnings == []

    def test_source_missing_ai_document_id_skipped_with_warning(self):
        """Source without ai_document_id must be skipped, not defaulted to 0."""
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        proposals = [
            {
                "status": "generated",
                "task_type": "change_proposal",
                "sources": [
                    {
                        # No ai_document_id key at all
                        "filename": "ghost.pdf",
                        "document_role": "other",
                    },
                ],
                "payload": {
                    "target_area": "Chuẩn đầu ra",
                    "change_type": "update",
                    "current_issue": "X",
                    "proposed_change": "Y",
                    "rationale": "Z",
                    "expected_impact": "W",
                    "priority": "high",
                    "confidence": "high",
                },
            },
        ]
        draft = _make_draft(change_proposals=proposals)
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert len(items) == 1
        assert items[0].sources == []  # ghost source skipped
        assert len(warnings) == 1
        assert "missing ai_document_id" in warnings[0].lower()

    def test_valid_source_still_preserved(self):
        """Source with ai_document_id is mapped normally."""
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        draft = _make_draft()  # default has ai_document_id=42
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert len(items[0].sources) == 1
        assert items[0].sources[0].ai_document_id == 42
        assert warnings == []

    def test_mixed_valid_and_invalid_sources(self):
        """Valid sources kept, invalid ones skipped with warning."""
        from app.api.v1.ctdt import _extract_mau06_items_from_draft

        proposals = [
            {
                "status": "generated",
                "task_type": "change_proposal",
                "sources": [
                    {"ai_document_id": 10, "filename": "valid.pdf"},
                    {"filename": "no_id.pdf"},  # missing ai_document_id
                    {"ai_document_id": 20, "filename": "also_valid.pdf"},
                ],
                "payload": {
                    "target_area": "Học phần",
                    "change_type": "add",
                    "current_issue": "",
                    "proposed_change": "",
                    "rationale": "",
                    "expected_impact": "",
                    "priority": "medium",
                    "confidence": "medium",
                },
            },
        ]
        draft = _make_draft(change_proposals=proposals)
        items, warnings = _extract_mau06_items_from_draft(draft)

        assert len(items) == 1
        assert len(items[0].sources) == 2
        assert items[0].sources[0].ai_document_id == 10
        assert items[0].sources[1].ai_document_id == 20
        assert len(warnings) == 1
        assert "source [1]" in warnings[0]


# ══════════════════════════════════════════════════════════════════════
# 2. Endpoint tests
# ══════════════════════════════════════════════════════════════════════


class TestMau06DraftEndpoint:
    """Test GET /update-cycles/{id}/mau-06/draft/latest."""

    @pytest.mark.asyncio
    async def test_returns_flat_items(self):
        from app.api.v1.ctdt import get_latest_mau06_draft

        draft = _make_draft()

        async def mock_get_latest(db, **kwargs):
            return draft

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            side_effect=mock_get_latest,
        ):
            response = await get_latest_mau06_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                program_code="7480201",
                analysis_mode="draft",
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        assert response.draft_id == 12
        assert response.update_cycle_id == "15"
        assert response.program_code == "7480201"
        assert response.program_name == "CNTT"
        assert response.analysis_mode == "draft"
        assert response.status == "draft"
        assert len(response.items) == 1
        assert response.items[0].proposed_change == "Bổ sung CĐR về AI"
        assert response.items[0].target_area == "Chuẩn đầu ra"
        assert len(response.items[0].sources) == 1
        assert response.items[0].sources[0].ai_document_id == 42
        assert response.warnings == []

    @pytest.mark.asyncio
    async def test_calls_get_latest_with_correct_tenant(self):
        """Verify endpoint passes tenant_id from current user."""
        from app.api.v1.ctdt import get_latest_mau06_draft

        draft = _make_draft()
        captured = {}

        async def mock_get_latest(db, **kwargs):
            captured.update(kwargs)
            return draft

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            side_effect=mock_get_latest,
        ):
            await get_latest_mau06_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                program_code="7480201",
                analysis_mode="draft",
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="tenant_abc", id=99),
            )

        assert captured["tenant_id"] == "tenant_abc"
        assert captured["update_cycle_id"] == "15"
        assert captured["program_code"] == "7480201"
        assert captured["analysis_mode"] == "draft"

    @pytest.mark.asyncio
    async def test_no_draft_returns_404(self):
        from app.api.v1.ctdt import get_latest_mau06_draft

        async def mock_get_latest(db, **kwargs):
            return None

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            side_effect=mock_get_latest,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_latest_mau06_draft(
                    update_cycle_id="15",
                    request=AsyncMock(),
                    program_code="7480201",
                    analysis_mode="draft",
                    db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["code"] == "mau06_draft_not_found"

    @pytest.mark.asyncio
    async def test_skeleton_mode_returns_422(self):
        from app.api.v1.ctdt import get_latest_mau06_draft

        with pytest.raises(HTTPException) as exc_info:
            await get_latest_mau06_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                program_code="7480201",
                analysis_mode="skeleton",
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "mau06_requires_draft_mode"

    @pytest.mark.asyncio
    async def test_unknown_mode_returns_422(self):
        from app.api.v1.ctdt import get_latest_mau06_draft

        with pytest.raises(HTTPException) as exc_info:
            await get_latest_mau06_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                analysis_mode="full",
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "mau06_requires_draft_mode"

    @pytest.mark.asyncio
    async def test_draft_with_no_change_proposals_returns_empty_items(self):
        """If the draft exists but has no change_proposals, return items=[] with metadata."""
        from app.api.v1.ctdt import get_latest_mau06_draft

        draft = _make_draft(change_proposals=[])

        async def mock_get_latest(db, **kwargs):
            return draft

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            side_effect=mock_get_latest,
        ):
            response = await get_latest_mau06_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                analysis_mode="draft",
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        assert response.draft_id == 12
        assert response.items == []
        assert response.warnings == []

    @pytest.mark.asyncio
    async def test_draft_with_missing_payload_returns_warnings(self):
        """Items without payload are skipped, warning included."""
        from app.api.v1.ctdt import get_latest_mau06_draft

        proposals = [
            {"status": "generated", "task_type": "change_proposal", "sources": []},
        ]
        draft = _make_draft(change_proposals=proposals)

        async def mock_get_latest(db, **kwargs):
            return draft

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            side_effect=mock_get_latest,
        ):
            response = await get_latest_mau06_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                analysis_mode="draft",
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        assert response.items == []
        assert len(response.warnings) == 1
        assert "missing payload" in response.warnings[0].lower()

    @pytest.mark.asyncio
    async def test_source_summary_passed_through(self):
        from app.api.v1.ctdt import get_latest_mau06_draft

        draft = _make_draft()

        async def mock_get_latest(db, **kwargs):
            return draft

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            side_effect=mock_get_latest,
        ):
            response = await get_latest_mau06_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                analysis_mode="draft",
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        # source_summary may be a Pydantic model or dict depending on serialization
        ss = response.source_summary
        if hasattr(ss, "contexts_count"):
            assert ss.contexts_count == 1
            assert ss.documents_used == [42]
            assert ss.tasks_executed == ["change_proposal"]
        else:
            assert ss["contexts_count"] == 1
            assert ss["documents_used"] == [42]
            assert ss["tasks_executed"] == ["change_proposal"]


# ══════════════════════════════════════════════════════════════════════
# 3. list_analysis_drafts status filter hardening
# ══════════════════════════════════════════════════════════════════════


class FakeListScalarResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeListExecuteResult:
    def __init__(self, rows):
        self.rows = rows

    def scalars(self):
        return FakeListScalarResult(self.rows)


class ListExecuteDB:
    """Fake DB that captures statement and returns rows."""
    def __init__(self, rows):
        self.rows = rows
        self.statement = None

    async def execute(self, stmt):
        self.statement = stmt
        return FakeListExecuteResult(self.rows)


class TestListAnalysisDraftsStatusFilter:
    """Ensure list_analysis_drafts defaults to status='draft'."""

    @pytest.mark.asyncio
    async def test_default_status_filters_draft(self):
        from app.services.ctdt_analysis_draft_service import list_analysis_drafts

        db = ListExecuteDB([])
        await list_analysis_drafts(db, tenant_id="t1")

        statement_text = str(db.statement)
        assert "ctdt_analysis_drafts.status" in statement_text

    @pytest.mark.asyncio
    async def test_status_none_no_status_filter(self):
        from app.services.ctdt_analysis_draft_service import list_analysis_drafts

        db = ListExecuteDB([])
        await list_analysis_drafts(db, tenant_id="t1", status=None)

        statement_text = str(db.statement)
        # When status=None, the compiled SQL should not contain a status equality filter
        # We check that the generated SQL doesn't filter on status
        # Note: the table name will appear in FROM clause; we check for the
        # specific pattern of "status = :status" or ".status ="
        where_parts = statement_text.split("WHERE")
        if len(where_parts) > 1:
            where_clause = where_parts[1]
            # status should NOT appear in WHERE clause
            assert "ctdt_analysis_drafts.status" not in where_clause

    @pytest.mark.asyncio
    async def test_explicit_status_archived(self):
        from app.services.ctdt_analysis_draft_service import list_analysis_drafts

        db = ListExecuteDB([])
        await list_analysis_drafts(db, tenant_id="t1", status="archived")

        statement_text = str(db.statement)
        assert "ctdt_analysis_drafts.status" in statement_text


# ══════════════════════════════════════════════════════════════════════
# 4. Existing analysis-drafts/latest contract preserved
# ══════════════════════════════════════════════════════════════════════


class TestExistingDraftEndpointUnchanged:
    """Verify the R5.5 endpoint still works unchanged."""

    @pytest.mark.asyncio
    async def test_analysis_drafts_latest_returns_result_payload(self):
        """Existing endpoint returns full result_payload, not flat items."""
        from app.api.v1.ctdt import get_latest_analysis_draft as api_get_latest

        draft = _make_draft()

        async def mock_get_latest(db, **kwargs):
            return draft

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            side_effect=mock_get_latest,
        ):
            response = await api_get_latest(
                update_cycle_id="15",
                request=AsyncMock(),
                program_code="7480201",
                analysis_mode="draft",
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        # R5.5 endpoint returns result_payload (not items)
        assert hasattr(response, "result_payload")
        assert "change_proposals" in response.result_payload
        # R5.5 does NOT have 'items' field
        assert not hasattr(response, "items")
