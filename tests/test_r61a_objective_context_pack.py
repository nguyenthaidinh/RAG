"""Tests for R6.1A Objective Update Context Pack."""
from __future__ import annotations

import inspect
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from app.services.ctdt_retrieval_service import (
    CTDTRetrievalContext,
    CTDTRetrievalResult,
    CTDTTaskType,
)


# ── Fixture helpers ──────────────────────────────────────────────────


def _make_ctx(
    *,
    ai_document_id: int = 1,
    document_role: str = "current_curriculum",
    filename: str = "ctdt.pdf",
    score: float = 0.85,
    text: str = "Mục tiêu đào tạo hiện hành",
    chunk_index: int = 0,
) -> CTDTRetrievalContext:
    return CTDTRetrievalContext(
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


def _make_retrieval_result(
    contexts: list[CTDTRetrievalContext] | None = None,
    roles: list[str] | None = None,
    scoped_document_count: int | None = None,
) -> CTDTRetrievalResult:
    if contexts is None:
        contexts = []
    return CTDTRetrievalResult(
        query="test",
        update_cycle_id="15",
        program_code="7480201",
        task_type="objective_suggestion",
        document_roles_used=roles or [],
        contexts=contexts,
        scoped_document_count=scoped_document_count if scoped_document_count is not None else len(contexts),
        latency_ms=10,
    )


# Track which role groups were retrieved
def _make_role_aware_mock():
    """Create a ctdt_retrieve mock that returns contexts based on document_roles."""
    calls = []

    async def mock_retrieve(db, **kwargs):
        roles = kwargs.get("document_roles", [])
        calls.append({"roles": roles, "query": kwargs.get("query", "")})

        if "current_curriculum" in roles:
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=1, document_role="current_curriculum")],
                roles=roles,
            )
        elif "direction_decision" in roles:
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=2, document_role="direction_decision",
                                    filename="quyet_dinh.pdf")],
                roles=roles,
            )
        elif "legal_regulation" in roles:
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=3, document_role="legal_regulation",
                                    filename="quy_dinh.pdf")],
                roles=roles,
            )
        elif "survey_evidence" in roles or "meeting_report" in roles:
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=4, document_role="survey_evidence",
                                    filename="khaosat.pdf")],
                roles=roles,
            )
        elif "comparison_report" in roles:
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=5, document_role="comparison_report",
                                    filename="doi_sanh.pdf")],
                roles=roles,
            )
        return _make_retrieval_result(contexts=[], roles=roles)

    return mock_retrieve, calls


# ══════════════════════════════════════════════════════════════════════
# 1. build context pack gọi retrieval theo nhiều role group
# ══════════════════════════════════════════════════════════════════════


class TestMultiRoleRetrieval:
    @pytest.mark.asyncio
    async def test_builds_context_pack_with_multiple_role_groups(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        mock_retrieve, calls = _make_role_aware_mock()

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
                program_code="7480201",
            )

        # R6.5: multi-query means more calls per group (especially current_objective)
        # The important invariant is that all 5 role groups were queried.
        assert len(calls) >= 5
        # Verify role groups called
        all_roles = [c["roles"] for c in calls]
        assert ["current_curriculum"] in all_roles
        assert ["direction_decision"] in all_roles
        assert ["legal_regulation"] in all_roles
        assert ["comparison_report"] in all_roles
        # survey_evidence + meeting_report in same group
        assert any("survey_evidence" in r for r in all_roles)


# ══════════════════════════════════════════════════════════════════════
# 2. current_curriculum → current_objective_contexts
# ══════════════════════════════════════════════════════════════════════


class TestCurrentCurriculumContexts:
    @pytest.mark.asyncio
    async def test_current_curriculum_in_current_objective_contexts(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        mock_retrieve, _ = _make_role_aware_mock()

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert len(pack.current_objective_contexts) >= 1
        assert pack.current_objective_contexts[0].document_role == "current_curriculum"


# ══════════════════════════════════════════════════════════════════════
# 3. direction_decision → direction_contexts
# ══════════════════════════════════════════════════════════════════════


class TestDirectionContexts:
    @pytest.mark.asyncio
    async def test_direction_decision_in_direction_contexts(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        mock_retrieve, _ = _make_role_aware_mock()

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert len(pack.direction_contexts) >= 1
        assert pack.direction_contexts[0].document_role == "direction_decision"


# ══════════════════════════════════════════════════════════════════════
# 4. thiếu current_curriculum → missing_information current_objectives
# ══════════════════════════════════════════════════════════════════════


class TestMissingCurrentCurriculum:
    @pytest.mark.asyncio
    async def test_missing_current_curriculum_flagged(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        async def mock_retrieve_empty(db, **kwargs):
            roles = kwargs.get("document_roles", [])
            # current_curriculum returns empty
            if "current_curriculum" in roles:
                return _make_retrieval_result(contexts=[], roles=roles)
            # Others return data
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=99, document_role=roles[0] if roles else "other")],
                roles=roles,
            )

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve_empty,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        mi_types = [m["type"] for m in pack.missing_information]
        assert "current_objectives" in mi_types
        assert pack.role_coverage["current_objective"].status == "missing"


# ══════════════════════════════════════════════════════════════════════
# 5. thiếu direction_decision → missing_information direction_decision
# ══════════════════════════════════════════════════════════════════════


class TestMissingDirectionDecision:
    @pytest.mark.asyncio
    async def test_missing_direction_decision_flagged(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        async def mock_retrieve_no_direction(db, **kwargs):
            roles = kwargs.get("document_roles", [])
            if "direction_decision" in roles:
                return _make_retrieval_result(contexts=[], roles=roles)
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=99, document_role=roles[0] if roles else "other")],
                roles=roles,
            )

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve_no_direction,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        mi_types = [m["type"] for m in pack.missing_information]
        assert "direction_decision" in mi_types
        assert pack.role_coverage["direction"].status == "missing"


# ══════════════════════════════════════════════════════════════════════
# 6. documents_used unique
# ══════════════════════════════════════════════════════════════════════


class TestDocumentsUsedUnique:
    @pytest.mark.asyncio
    async def test_documents_used_unique_in_source_summary(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        mock_retrieve, _ = _make_role_aware_mock()

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        docs = pack.source_summary.documents_used
        assert len(docs) == len(set(docs)), "documents_used must be unique"
        assert docs == sorted(docs), "documents_used must be sorted"


# ══════════════════════════════════════════════════════════════════════
# 7. endpoint context-pack không gọi LLM
# ══════════════════════════════════════════════════════════════════════


class TestEndpointNoLLM:
    @pytest.mark.asyncio
    async def test_endpoint_does_not_call_llm(self):
        from app.api.v1.ctdt import (
            ObjectiveContextPackRequest,
            build_objective_context_pack,
        )
        from app.services.ctdt_objective_context_service import (
            ObjectiveUpdateContextPack,
            ContextPackSourceSummary,
        )

        pack = ObjectiveUpdateContextPack(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
        )
        pack.source_summary = ContextPackSourceSummary(
            total_contexts=0,
            documents_used=[],
            role_groups_retrieved=["current_objective"],
            latency_ms=10,
        )

        async def mock_build(db, **kwargs):
            return pack

        with patch(
            "app.services.ctdt_objective_context_service.build_objective_update_context_pack",
            side_effect=mock_build,
        ):
            response = await build_objective_context_pack(
                body=ObjectiveContextPackRequest(
                    update_cycle_id="15",
                    program_code="7480201",
                ),
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        assert response.context_pack_type == "objective_update"
        assert response.update_cycle_id == "15"

    def test_service_has_no_llm_imports(self):
        """Verify the service module does not import LLM-related modules."""
        import app.services.ctdt_objective_context_service as mod

        source = inspect.getsource(mod)
        import_lines = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith(("import ", "from "))
        ]
        joined = "\n".join(import_lines)
        assert "openai" not in joined.lower()
        assert "httpx" not in joined.lower()
        assert "AnswerService" not in joined


# ══════════════════════════════════════════════════════════════════════
# 8. endpoint context-pack không commit DB
# ══════════════════════════════════════════════════════════════════════


class TestEndpointNoDBCommit:
    @pytest.mark.asyncio
    async def test_endpoint_no_db_commit(self):
        from app.api.v1.ctdt import (
            ObjectiveContextPackRequest,
            build_objective_context_pack,
        )
        from app.services.ctdt_objective_context_service import (
            ObjectiveUpdateContextPack,
            ContextPackSourceSummary,
        )

        pack = ObjectiveUpdateContextPack(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
        )
        pack.source_summary = ContextPackSourceSummary(
            total_contexts=0,
            documents_used=[],
            role_groups_retrieved=[],
            latency_ms=10,
        )

        async def mock_build(db, **kwargs):
            return pack

        db = AsyncMock()
        with patch(
            "app.services.ctdt_objective_context_service.build_objective_update_context_pack",
            side_effect=mock_build,
        ):
            await build_objective_context_pack(
                body=ObjectiveContextPackRequest(update_cycle_id="15"),
                request=AsyncMock(),
                db=db,
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        db.add.assert_not_called()
        db.commit.assert_not_called()
        db.flush.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# 9. không ghi Program/ProgramVersion/ProgramVersionRevision
# ══════════════════════════════════════════════════════════════════════


class TestNoProgramWrites:
    def test_service_has_no_program_model_imports(self):
        import app.services.ctdt_objective_context_service as mod

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
# 10. Role coverage full scenario
# ══════════════════════════════════════════════════════════════════════


class TestRoleCoverageFull:
    @pytest.mark.asyncio
    async def test_full_coverage_all_available(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        mock_retrieve, _ = _make_role_aware_mock()

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        # All 5 groups should be available
        for key in ("current_objective", "direction", "legal", "evidence", "comparison"):
            assert pack.role_coverage[key].status == "available", f"{key} should be available"
            assert pack.role_coverage[key].context_count >= 1

        # No missing information
        assert pack.missing_information == []

    @pytest.mark.asyncio
    async def test_all_empty_returns_all_missing(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        async def mock_empty(db, **kwargs):
            return _make_retrieval_result(contexts=[])

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_empty,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        mi_types = [m["type"] for m in pack.missing_information]
        assert "current_objectives" in mi_types
        assert "direction_decision" in mi_types
        assert "legal_regulation" in mi_types
        assert "survey_evidence" in mi_types
        assert "comparison_report" in mi_types

        for key in pack.role_coverage:
            assert pack.role_coverage[key].status == "missing"


# ══════════════════════════════════════════════════════════════════════
# 11. Error handling
# ══════════════════════════════════════════════════════════════════════


class TestEndpointErrorHandling:
    @pytest.mark.asyncio
    async def test_service_error_returns_500(self):
        from app.api.v1.ctdt import (
            ObjectiveContextPackRequest,
            build_objective_context_pack,
        )

        async def mock_build(db, **kwargs):
            raise RuntimeError("retrieval exploded")

        with patch(
            "app.services.ctdt_objective_context_service.build_objective_update_context_pack",
            side_effect=mock_build,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await build_objective_context_pack(
                    body=ObjectiveContextPackRequest(update_cycle_id="15"),
                    request=AsyncMock(),
                    db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                    query_svc=AsyncMock(),
                )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "objective_context_pack_error"


# ══════════════════════════════════════════════════════════════════════
# 12. Retrieval failure for one group doesn't crash others
# ══════════════════════════════════════════════════════════════════════


class TestPartialRetrievalFailure:
    @pytest.mark.asyncio
    async def test_one_group_failure_does_not_crash(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        call_count = 0

        async def mock_retrieve_with_failure(db, **kwargs):
            nonlocal call_count
            call_count += 1
            roles = kwargs.get("document_roles", [])
            if "legal_regulation" in roles:
                raise RuntimeError("DB connection lost")
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=call_count, document_role=roles[0] if roles else "other")],
                roles=roles,
            )

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve_with_failure,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        # legal group failed → failed (not missing)
        assert pack.role_coverage["legal"].status == "failed"
        assert pack.role_coverage["legal"].retrieval_status == "failed"
        # Other groups should still have contexts
        assert pack.role_coverage["current_objective"].status == "available"
        assert pack.role_coverage["current_objective"].retrieval_status == "ok"
        assert pack.source_summary.total_contexts >= 1


# ══════════════════════════════════════════════════════════════════════
# 13. document_available_no_context status
# ══════════════════════════════════════════════════════════════════════


class TestDocumentAvailableNoContext:
    @pytest.mark.asyncio
    async def test_scoped_docs_but_no_contexts_returns_document_available_no_context(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        async def mock_retrieve(db, **kwargs):
            roles = kwargs.get("document_roles", [])
            if "current_curriculum" in roles:
                # Has scoped documents but no matching contexts
                return _make_retrieval_result(
                    contexts=[], roles=roles, scoped_document_count=3,
                )
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=99, document_role=roles[0] if roles else "other")],
                roles=roles,
            )

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        cov = pack.role_coverage["current_objective"]
        assert cov.status == "document_available_no_context"
        assert cov.scoped_document_count == 3
        assert cov.context_count == 0
        assert cov.retrieval_status == "ok"

    @pytest.mark.asyncio
    async def test_scoped_0_contexts_0_returns_missing(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        async def mock_empty(db, **kwargs):
            return _make_retrieval_result(
                contexts=[], scoped_document_count=0,
            )

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_empty,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        for key in pack.role_coverage:
            assert pack.role_coverage[key].status == "missing"
            assert pack.role_coverage[key].scoped_document_count == 0


# ══════════════════════════════════════════════════════════════════════
# 14. context_not_found vs missing missing_information
# ══════════════════════════════════════════════════════════════════════


class TestMissingInfoDistinction:
    @pytest.mark.asyncio
    async def test_missing_produces_current_objectives_type(self):
        """status=missing → type=current_objectives (no documents at all)."""
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(contexts=[], scoped_document_count=0)

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        mi_types = [m["type"] for m in pack.missing_information]
        assert "current_objectives" in mi_types
        assert "current_objectives_context_not_found" not in mi_types

    @pytest.mark.asyncio
    async def test_doc_available_no_context_produces_context_not_found_type(self):
        """status=document_available_no_context → type=current_objectives_context_not_found."""
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        async def mock_retrieve(db, **kwargs):
            roles = kwargs.get("document_roles", [])
            if "current_curriculum" in roles:
                return _make_retrieval_result(
                    contexts=[], roles=roles, scoped_document_count=2,
                )
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=99, document_role=roles[0] if roles else "other")],
                roles=roles,
            )

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        mi_types = [m["type"] for m in pack.missing_information]
        assert "current_objectives_context_not_found" in mi_types
        assert "current_objectives" not in mi_types  # not "missing", it's "context_not_found"


# ══════════════════════════════════════════════════════════════════════
# 15. Retrieval failure → status=failed, retrieval_status=failed
# ══════════════════════════════════════════════════════════════════════


class TestRetrievalFailedStatus:
    @pytest.mark.asyncio
    async def test_retrieval_exception_sets_failed(self):
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )

        async def mock_retrieve(db, **kwargs):
            roles = kwargs.get("document_roles", [])
            if "direction_decision" in roles:
                raise RuntimeError("connection refused")
            return _make_retrieval_result(
                contexts=[_make_ctx(ai_document_id=1, document_role=roles[0] if roles else "other")],
                roles=roles,
            )

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        direction_cov = pack.role_coverage["direction"]
        assert direction_cov.status == "failed"
        assert direction_cov.retrieval_status == "failed"
        assert direction_cov.scoped_document_count == 0
        assert direction_cov.context_count == 0

        # failed group should NOT add missing_information
        # (we don't know if documents exist, just that retrieval failed)
        mi_types = [m["type"] for m in pack.missing_information]
        assert "direction_decision" not in mi_types
        assert "direction_decision_context_not_found" not in mi_types


# ══════════════════════════════════════════════════════════════════════
# 16. Endpoint maps scoped_document_count and retrieval_status
# ══════════════════════════════════════════════════════════════════════


class TestEndpointRoleCoverageMapping:
    @pytest.mark.asyncio
    async def test_endpoint_maps_role_coverage_extra_fields(self):
        """Verify scoped_document_count and retrieval_status reach API response."""
        from app.api.v1.ctdt import (
            ObjectiveContextPackRequest,
            build_objective_context_pack,
        )
        from app.services.ctdt_objective_context_service import (
            ContextPackSourceSummary,
            ObjectiveUpdateContextPack,
            RoleCoverageItem,
        )

        pack = ObjectiveUpdateContextPack(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
        )
        pack.role_coverage["current_objective"] = RoleCoverageItem(
            document_roles=["current_curriculum"],
            context_count=0,
            documents_used=[],
            status="document_available_no_context",
            scoped_document_count=3,
            retrieval_status="ok",
        )
        pack.role_coverage["direction"] = RoleCoverageItem(
            document_roles=["direction_decision"],
            context_count=0,
            documents_used=[],
            status="failed",
            scoped_document_count=0,
            retrieval_status="failed",
        )
        pack.source_summary = ContextPackSourceSummary(
            total_contexts=0,
            documents_used=[],
            role_groups_retrieved=["current_objective", "direction"],
            latency_ms=10,
        )

        async def mock_build(db, **kwargs):
            return pack

        with patch(
            "app.services.ctdt_objective_context_service.build_objective_update_context_pack",
            side_effect=mock_build,
        ):
            response = await build_objective_context_pack(
                body=ObjectiveContextPackRequest(
                    update_cycle_id="15",
                    program_code="7480201",
                ),
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        # document_available_no_context case
        co = response.role_coverage["current_objective"]
        assert co.status == "document_available_no_context"
        assert co.scoped_document_count == 3
        assert co.retrieval_status == "ok"

        # failed case
        dr = response.role_coverage["direction"]
        assert dr.status == "failed"
        assert dr.scoped_document_count == 0
        assert dr.retrieval_status == "failed"
