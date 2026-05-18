"""Tests for R5.5 CTDT analysis draft persistence."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.db.models.ctdt_analysis_draft import CTDTAnalysisDraft
from app.services.ctdt_analysis_draft_service import (
    get_latest_analysis_draft,
    save_analysis_draft,
)
from app.services.ctdt_analysis_service import (
    AnalysisCycleResult,
    AnalysisSkeletonItem,
    AnalysisSourceSummary,
)


def _make_analysis_result(*, proposed_change: str = "Bo sung CDR ve AI") -> AnalysisCycleResult:
    return AnalysisCycleResult(
        update_cycle_id="15",
        program_code="7480201",
        program_name="CNTT",
        analysis_mode="draft",
        result_payload={
            "change_proposals": [
                AnalysisSkeletonItem(
                    status="generated",
                    task_type="change_proposal",
                    sources=[],
                    payload={
                        "target_area": "Chuan dau ra",
                        "change_type": "update",
                        "current_issue": "CDR chua phan anh nang luc AI",
                        "proposed_change": proposed_change,
                        "rationale": "Co minh chung tu tai lieu cap nhat",
                        "expected_impact": "Sinh vien dap ung tot hon yeu cau viec lam",
                        "priority": "high",
                        "confidence": "high",
                    },
                )
            ],
        },
        source_summary=AnalysisSourceSummary(
            contexts_count=1,
            documents_used=[42],
            tasks_executed=["change_proposal"],
            latency_ms=25,
        ),
    )


class FakeDraftDB:
    def __init__(self):
        self.rows = []
        self.add_calls = 0
        self.flush_calls = 0
        self.refresh_calls = 0
        self.commit = AsyncMock()
        self.rollback = AsyncMock()

    def add(self, row):
        self.add_calls += 1
        row.id = len(self.rows) + 1
        now = datetime.now(timezone.utc)
        row.created_at = now
        row.updated_at = now
        self.rows.append(row)

    async def flush(self):
        self.flush_calls += 1

    async def refresh(self, row):
        self.refresh_calls += 1


class FakeScalarResult:
    def __init__(self, row):
        self.row = row

    def first(self):
        return self.row


class FakeExecuteResult:
    def __init__(self, row):
        self.row = row

    def scalars(self):
        return FakeScalarResult(self.row)


class ExecuteOnlyDB:
    def __init__(self, row):
        self.row = row
        self.statement = None

    async def execute(self, stmt):
        self.statement = stmt
        return FakeExecuteResult(self.row)


class StatusAwareExecuteDB:
    def __init__(self, *, active_row, archived_row):
        self.active_row = active_row
        self.archived_row = archived_row
        self.statement = None

    async def execute(self, stmt):
        self.statement = stmt
        statement_text = str(stmt)
        if "ctdt_analysis_drafts.status" in statement_text:
            return FakeExecuteResult(self.active_row)
        return FakeExecuteResult(self.archived_row)


class TestAnalysisDraftService:
    @pytest.mark.asyncio
    async def test_save_analysis_draft_persists_payload_without_commit(self):
        db = FakeDraftDB()
        draft = await save_analysis_draft(
            db,
            tenant_id="t1",
            user_id=7,
            result=_make_analysis_result(),
            program_id="p1",
        )

        assert draft.id == 1
        assert draft.tenant_id == "t1"
        assert draft.program_id == "p1"
        assert draft.result_payload["change_proposals"][0]["payload"]["proposed_change"] == "Bo sung CDR ve AI"
        assert draft.source_summary["documents_used"] == [42]
        assert db.add_calls == 1
        assert db.flush_calls == 1
        assert db.refresh_calls == 1
        db.commit.assert_not_called()
        assert all(row.__tablename__ == "ctdt_analysis_drafts" for row in db.rows)

    @pytest.mark.asyncio
    async def test_get_latest_analysis_draft_uses_latest_ordering(self):
        newer = CTDTAnalysisDraft(
            tenant_id="t1",
            update_cycle_id="15",
            program_code="7480201",
            analysis_mode="draft",
            draft_type="update_cycle_analysis",
            result_payload={"change_proposals": []},
            source_summary={"contexts_count": 0, "documents_used": [], "tasks_executed": [], "latency_ms": 0},
            status="draft",
        )
        newer.id = 2
        newer.created_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        newer.updated_at = datetime.now(timezone.utc)

        db = ExecuteOnlyDB(newer)
        draft = await get_latest_analysis_draft(
            db,
            tenant_id="t1",
            update_cycle_id="15",
            program_code="7480201",
            analysis_mode="draft",
        )

        assert draft is newer
        statement_text = str(db.statement)
        assert "ctdt_analysis_drafts.updated_at DESC" in statement_text
        assert "ctdt_analysis_drafts.id DESC" in statement_text
        assert "ctdt_analysis_drafts.status" in statement_text

    @pytest.mark.asyncio
    async def test_get_latest_analysis_draft_ignores_archived_newer_draft(self):
        active = CTDTAnalysisDraft(
            tenant_id="t1",
            update_cycle_id="15",
            program_code="7480201",
            analysis_mode="draft",
            draft_type="update_cycle_analysis",
            result_payload={"change_proposals": []},
            source_summary={"contexts_count": 0, "documents_used": [], "tasks_executed": [], "latency_ms": 0},
            status="draft",
        )
        active.id = 1
        active.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
        active.updated_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        archived = CTDTAnalysisDraft(
            tenant_id="t1",
            update_cycle_id="15",
            program_code="7480201",
            analysis_mode="draft",
            draft_type="update_cycle_analysis",
            result_payload={"change_proposals": []},
            source_summary={"contexts_count": 0, "documents_used": [], "tasks_executed": [], "latency_ms": 0},
            status="archived",
        )
        archived.id = 2
        archived.created_at = datetime.now(timezone.utc) - timedelta(minutes=2)
        archived.updated_at = datetime.now(timezone.utc)

        db = StatusAwareExecuteDB(active_row=active, archived_row=archived)
        draft = await get_latest_analysis_draft(
            db,
            tenant_id="t1",
            update_cycle_id="15",
            program_code="7480201",
            analysis_mode="draft",
        )

        assert draft is active
        assert draft.status == "draft"


class TestAnalyzeDraftSaveAPI:
    @pytest.mark.asyncio
    async def test_save_draft_false_does_not_write_db(self):
        from app.api.v1.ctdt import (
            AnalyzeUpdateCycleRequest,
            analyze_update_cycle as api_analyze_update_cycle,
        )

        async def mock_do_analyze(db, **kwargs):
            return _make_analysis_result()

        db = AsyncMock()
        with patch(
            "app.services.ctdt_analysis_service.analyze_update_cycle",
            side_effect=mock_do_analyze,
        ):
            response = await api_analyze_update_cycle(
                body=AnalyzeUpdateCycleRequest(
                    update_cycle_id="15",
                    program_code="7480201",
                    analysis_mode="draft",
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

    @pytest.mark.asyncio
    async def test_save_draft_true_persists_draft(self):
        from app.api.v1.ctdt import (
            AnalyzeUpdateCycleRequest,
            analyze_update_cycle as api_analyze_update_cycle,
        )

        async def mock_do_analyze(db, **kwargs):
            return _make_analysis_result(
                proposed_change="Bo sung CDR ve ung dung AI co ban"
            )

        db = FakeDraftDB()
        with patch(
            "app.services.ctdt_analysis_service.analyze_update_cycle",
            side_effect=mock_do_analyze,
        ):
            response = await api_analyze_update_cycle(
                body=AnalyzeUpdateCycleRequest(
                    update_cycle_id="15",
                    program_code="7480201",
                    program_id="p1",
                    analysis_mode="draft",
                    save_draft=True,
                ),
                request=AsyncMock(),
                db=db,
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        assert response.draft_saved is True
        assert response.draft_id == 1
        assert db.add_calls == 1
        assert db.flush_calls == 1
        assert db.refresh_calls == 1
        db.commit.assert_awaited_once()
        db.rollback.assert_not_called()
        assert db.rows[0].result_payload["change_proposals"][0]["payload"]["proposed_change"] == (
            "Bo sung CDR ve ung dung AI co ban"
        )
        assert all(row.__tablename__ == "ctdt_analysis_drafts" for row in db.rows)

    @pytest.mark.asyncio
    async def test_save_draft_true_save_error_rolls_back(self):
        from app.api.v1.ctdt import (
            AnalyzeUpdateCycleRequest,
            analyze_update_cycle as api_analyze_update_cycle,
        )

        async def mock_do_analyze(db, **kwargs):
            return _make_analysis_result()

        async def mock_save_analysis_draft(db, **kwargs):
            raise RuntimeError("save failed")

        db = AsyncMock()
        with patch(
            "app.services.ctdt_analysis_service.analyze_update_cycle",
            side_effect=mock_do_analyze,
        ):
            with patch(
                "app.services.ctdt_analysis_draft_service.save_analysis_draft",
                side_effect=mock_save_analysis_draft,
            ):
                with pytest.raises(HTTPException) as exc_info:
                    await api_analyze_update_cycle(
                        body=AnalyzeUpdateCycleRequest(
                            update_cycle_id="15",
                            program_code="7480201",
                            analysis_mode="draft",
                            save_draft=True,
                        ),
                        request=AsyncMock(),
                        db=db,
                        user=SimpleNamespace(tenant_id="t1", id=7),
                        query_svc=AsyncMock(),
                    )

        assert exc_info.value.status_code == 500
        db.rollback.assert_awaited_once()
        db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_save_draft_true_skeleton_is_rejected(self):
        from app.api.v1.ctdt import (
            AnalyzeUpdateCycleRequest,
            analyze_update_cycle as api_analyze_update_cycle,
        )

        mock_do_analyze = AsyncMock()
        with patch(
            "app.services.ctdt_analysis_service.analyze_update_cycle",
            mock_do_analyze,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await api_analyze_update_cycle(
                    body=AnalyzeUpdateCycleRequest(
                        update_cycle_id="15",
                        analysis_mode="skeleton",
                        save_draft=True,
                    ),
                    request=AsyncMock(),
                    db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                    query_svc=AsyncMock(),
                )

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "draft_save_requires_draft_mode"
        mock_do_analyze.assert_not_called()


class TestLatestDraftAPI:
    @pytest.mark.asyncio
    async def test_get_latest_draft_returns_saved_payload_for_tenant(self):
        from app.api.v1.ctdt import get_latest_analysis_draft as api_get_latest

        draft = CTDTAnalysisDraft(
            tenant_id="t1",
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
            analysis_mode="draft",
            draft_type="update_cycle_analysis",
            result_payload={
                "change_proposals": [
                    {
                        "status": "generated",
                        "task_type": "change_proposal",
                        "sources": [],
                        "payload": {"proposed_change": "Bo sung CDR ve AI"},
                    }
                ],
            },
            source_summary={
                "contexts_count": 1,
                "documents_used": [42],
                "tasks_executed": ["change_proposal"],
                "latency_ms": 25,
            },
            status="draft",
        )
        draft.id = 2
        draft.created_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        draft.updated_at = datetime.now(timezone.utc)

        captured = {}

        async def mock_get_latest(db, **kwargs):
            captured.update(kwargs)
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

        assert captured["tenant_id"] == "t1"
        assert response.draft_id == 2
        assert response.result_payload["change_proposals"][0].payload["proposed_change"] == "Bo sung CDR ve AI"
