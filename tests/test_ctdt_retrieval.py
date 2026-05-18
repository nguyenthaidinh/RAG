"""
Tests for R3 — CTĐT Metadata Retrieval + Document Role Intelligence.

Tests:
1. Role policy resolution (task_type → document_roles)
2. Explicit roles override policy
3. Update cycle isolation (cycle 15 ≠ cycle 16)
4. Document role filtering
5. No results → empty contexts (no fallback)
6. Source metadata enrichment
7. Defense-in-depth scope filter
8. Task type validation
9. Backward compatibility (old endpoints unaffected)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from app.services.ctdt_retrieval_service import (
    CTDTTaskType,
    CTDTRetrievalContext,
    CTDTRetrievalResult,
    TASK_ROLE_POLICY,
    resolve_document_roles,
    _enrich_results,
    _get_scoped_document_ids,
)


# ── Role policy tests ────────────────────────────────────────────────


class TestResolveDocumentRoles:
    """Test task_type → document_roles policy resolution."""

    def test_evidence_analysis_policy(self):
        roles = resolve_document_roles(CTDTTaskType.EVIDENCE_ANALYSIS, None)
        assert "survey_evidence" in roles
        assert "meeting_report" in roles
        assert "direction_decision" in roles
        assert "legal_regulation" in roles
        assert "current_curriculum" not in roles

    def test_current_curriculum_review_policy(self):
        roles = resolve_document_roles(CTDTTaskType.CURRENT_CURRICULUM_REVIEW, None)
        assert "current_curriculum" in roles
        assert "comparison_report" in roles
        assert "legal_regulation" in roles
        assert "survey_evidence" not in roles

    def test_change_proposal_policy(self):
        roles = resolve_document_roles(CTDTTaskType.CHANGE_PROPOSAL, None)
        assert "current_curriculum" in roles
        assert "comparison_report" in roles
        assert "survey_evidence" in roles
        assert "meeting_report" in roles
        assert "legal_regulation" in roles

    def test_objective_suggestion_policy(self):
        roles = resolve_document_roles(CTDTTaskType.OBJECTIVE_SUGGESTION, None)
        assert "current_curriculum" in roles
        assert "survey_evidence" in roles

    def test_outcome_suggestion_policy(self):
        roles = resolve_document_roles(CTDTTaskType.OUTCOME_SUGGESTION, None)
        assert "current_curriculum" in roles
        assert "course_syllabus" in roles
        assert "legal_regulation" in roles

    def test_course_structure_policy(self):
        roles = resolve_document_roles(CTDTTaskType.COURSE_STRUCTURE, None)
        assert "current_curriculum" in roles
        assert "course_syllabus" in roles

    def test_matrix_mapping_policy(self):
        roles = resolve_document_roles(CTDTTaskType.MATRIX_MAPPING, None)
        assert "current_curriculum" in roles
        assert "course_syllabus" in roles
        assert len(roles) == 2

    def test_template_lookup_policy(self):
        roles = resolve_document_roles(CTDTTaskType.TEMPLATE_LOOKUP, None)
        assert "template" in roles
        assert "legal_regulation" in roles

    def test_general_query_no_role_filter(self):
        """general_query should return empty list (no role filter)."""
        roles = resolve_document_roles(CTDTTaskType.GENERAL_QUERY, None)
        assert roles == []

    def test_explicit_roles_override_policy(self):
        """Explicit roles should override task_type policy completely."""
        roles = resolve_document_roles(
            CTDTTaskType.EVIDENCE_ANALYSIS,
            ["current_curriculum"],
        )
        assert roles == ["current_curriculum"]
        assert "survey_evidence" not in roles

    def test_explicit_empty_list_uses_policy(self):
        """Empty explicit list should fall back to policy."""
        roles = resolve_document_roles(CTDTTaskType.EVIDENCE_ANALYSIS, [])
        assert "survey_evidence" in roles


class TestTaskTypeEnum:
    """Test CTDTTaskType enum values."""

    def test_all_task_types_have_policy(self):
        """Every task type should have an entry in TASK_ROLE_POLICY."""
        for task_type in CTDTTaskType:
            assert task_type in TASK_ROLE_POLICY, f"Missing policy for {task_type}"

    def test_task_type_values(self):
        assert CTDTTaskType.GENERAL_QUERY.value == "general_query"
        assert CTDTTaskType.EVIDENCE_ANALYSIS.value == "evidence_analysis"
        assert CTDTTaskType.CURRENT_CURRICULUM_REVIEW.value == "current_curriculum_review"
        assert CTDTTaskType.TEMPLATE_LOOKUP.value == "template_lookup"

    def test_invalid_task_type(self):
        with pytest.raises(ValueError):
            CTDTTaskType("invalid_task")


# ── Enrich results tests ─────────────────────────────────────────────


class TestEnrichResults:
    """Test result enrichment with CTĐT metadata."""

    def _make_query_result(self, document_id: int, score: float = 0.8):
        from app.services.retrieval.types import QueryResult
        return QueryResult(
            chunk_id=document_id * 100_000 + 5,
            document_id=document_id,
            score=score,
            snippet="Test snippet",
            highlights=(),
        )

    def test_enrichment_attaches_metadata(self):
        """Result should include CTĐT source metadata."""
        results = [self._make_query_result(42)]
        doc_meta = {
            42: {
                "external_file_id": "file_123",
                "filename": "Mau07_CTDT.docx",
                "document_role": "current_curriculum",
                "update_cycle_id": "15",
                "program_code": "7480201",
                "program_id": None,
            },
        }
        contexts = _enrich_results(results, doc_meta)
        assert len(contexts) == 1
        ctx = contexts[0]
        assert ctx.ai_document_id == 42
        assert ctx.external_file_id == "file_123"
        assert ctx.filename == "Mau07_CTDT.docx"
        assert ctx.document_role == "current_curriculum"
        assert ctx.source["update_cycle_id"] == "15"
        assert ctx.source["program_code"] == "7480201"
        assert ctx.chunk_index == 5

    def test_defense_in_depth_filters_out_of_scope(self):
        """Results from documents NOT in doc_metadata should be filtered out."""
        results = [
            self._make_query_result(42),  # in scope
            self._make_query_result(99),  # NOT in scope
        ]
        doc_meta = {
            42: {
                "external_file_id": "f1",
                "filename": "test.docx",
                "document_role": "current_curriculum",
                "update_cycle_id": "15",
                "program_code": "7480201",
                "program_id": None,
            },
        }
        contexts = _enrich_results(results, doc_meta)
        assert len(contexts) == 1
        assert contexts[0].ai_document_id == 42

    def test_empty_results(self):
        contexts = _enrich_results([], {})
        assert contexts == []

    def test_score_rounding(self):
        results = [self._make_query_result(1, score=0.123456789)]
        doc_meta = {1: {"external_file_id": "f1", "filename": "t.docx",
                        "document_role": "other", "update_cycle_id": "1",
                        "program_code": None, "program_id": None}}
        contexts = _enrich_results(results, doc_meta)
        assert contexts[0].score == 0.1235  # rounded to 4 decimal places


# ── Update cycle isolation tests ─────────────────────────────────────


class TestUpdateCycleIsolation:
    """Test that metadata scoping prevents cross-cycle leakage."""

    def _make_mock_doc(self, doc_id, update_cycle_id, document_role, program_code=None, status="ready"):
        """Create a mock Document object."""
        doc = MagicMock()
        doc.id = doc_id
        doc.tenant_id = "t1"
        doc.source = "cn_ctdt"
        doc.status = status
        doc.title = f"doc_{doc_id}.docx"
        doc.meta = {
            "ctdt": {
                "update_cycle_id": str(update_cycle_id),
                "document_role": document_role,
                "program_code": program_code,
                "program_id": None,
                "external_file_id": f"file_{doc_id}",
            }
        }
        return doc

    @pytest.mark.asyncio
    async def test_cycle_isolation(self):
        """Documents from cycle 16 must not appear in cycle 15 scope."""
        docs = [
            self._make_mock_doc(1, "15", "current_curriculum", "7480201"),
            self._make_mock_doc(2, "15", "survey_evidence", "7480201"),
            self._make_mock_doc(3, "16", "current_curriculum", "7480201"),  # different cycle
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        scoped = await _get_scoped_document_ids(
            mock_db,
            tenant_id="t1",
            update_cycle_id="15",
        )

        assert 1 in scoped
        assert 2 in scoped
        assert 3 not in scoped  # cycle 16 filtered out

    @pytest.mark.asyncio
    async def test_role_filter(self):
        """Only documents with matching roles should be returned."""
        docs = [
            self._make_mock_doc(1, "15", "current_curriculum"),
            self._make_mock_doc(2, "15", "survey_evidence"),
            self._make_mock_doc(3, "15", "meeting_report"),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        scoped = await _get_scoped_document_ids(
            mock_db,
            tenant_id="t1",
            update_cycle_id="15",
            document_roles=["current_curriculum"],
        )

        assert 1 in scoped
        assert 2 not in scoped
        assert 3 not in scoped

    @pytest.mark.asyncio
    async def test_program_code_filter(self):
        """Program code filter should narrow scope."""
        docs = [
            self._make_mock_doc(1, "15", "current_curriculum", "7480201"),
            self._make_mock_doc(2, "15", "current_curriculum", "7480202"),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        scoped = await _get_scoped_document_ids(
            mock_db,
            tenant_id="t1",
            update_cycle_id="15",
            program_code="7480201",
        )

        assert 1 in scoped
        assert 2 not in scoped

    @pytest.mark.asyncio
    async def test_no_matching_docs_returns_empty(self):
        """No matching documents should return empty dict."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        scoped = await _get_scoped_document_ids(
            mock_db,
            tenant_id="t1",
            update_cycle_id="999",
        )

        assert scoped == {}


# ── ctdt_retrieve integration tests ──────────────────────────────────


class TestCTDTRetrieveIntegration:
    """Test the main ctdt_retrieve function."""

    @pytest.mark.asyncio
    async def test_no_docs_returns_empty_contexts(self):
        """When no documents match scope, return empty contexts, no fallback."""
        from app.services.ctdt_retrieval_service import ctdt_retrieve

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await ctdt_retrieve(
            mock_db,
            tenant_id="t1",
            user_id=1,
            query="Test query",
            update_cycle_id="999",
            task_type=CTDTTaskType.GENERAL_QUERY,
        )

        assert result.contexts == []
        assert result.scoped_document_count == 0
        assert result.update_cycle_id == "999"
        assert result.task_type == "general_query"

    @pytest.mark.asyncio
    async def test_task_type_populates_roles_used(self):
        """task_type should resolve to document_roles_used in result."""
        from app.services.ctdt_retrieval_service import ctdt_retrieve

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await ctdt_retrieve(
            mock_db,
            tenant_id="t1",
            user_id=1,
            query="Test",
            update_cycle_id="15",
            task_type=CTDTTaskType.EVIDENCE_ANALYSIS,
        )

        assert "survey_evidence" in result.document_roles_used
        assert "meeting_report" in result.document_roles_used

    @pytest.mark.asyncio
    async def test_explicit_roles_override(self):
        """Explicit document_roles should override task_type policy."""
        from app.services.ctdt_retrieval_service import ctdt_retrieve

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await ctdt_retrieve(
            mock_db,
            tenant_id="t1",
            user_id=1,
            query="Test",
            update_cycle_id="15",
            task_type=CTDTTaskType.EVIDENCE_ANALYSIS,
            document_roles=["current_curriculum"],
        )

        assert result.document_roles_used == ["current_curriculum"]


# ── Schema validation tests ──────────────────────────────────────────


class TestCTDTRetrieveSchemas:
    """Test request/response schema validation."""

    def test_request_schema_basic(self):
        from app.api.v1.ctdt import CTDTRetrieveRequest
        req = CTDTRetrieveRequest(
            query="Danh sách học phần AI",
            update_cycle_id="15",
            program_code="7480201",
        )
        assert req.query == "Danh sách học phần AI"
        assert req.update_cycle_id == "15"
        assert req.task_type == "general_query"
        assert req.top_k == 8

    def test_request_schema_with_roles(self):
        from app.api.v1.ctdt import CTDTRetrieveRequest
        req = CTDTRetrieveRequest(
            query="Test",
            update_cycle_id="15",
            document_roles=["current_curriculum", "survey_evidence"],
        )
        assert req.document_roles == ["current_curriculum", "survey_evidence"]

    def test_response_schema(self):
        from app.api.v1.ctdt import (
            CTDTRetrieveResponse,
            CTDTRetrieveContextItem,
            CTDTRetrieveSourceMeta,
        )
        resp = CTDTRetrieveResponse(
            query="Test",
            update_cycle_id="15",
            task_type="general_query",
            document_roles_used=[],
            contexts=[
                CTDTRetrieveContextItem(
                    ai_document_id=42,
                    external_file_id="f1",
                    filename="test.docx",
                    document_role="current_curriculum",
                    chunk_id=4200005,
                    chunk_index=5,
                    score=0.85,
                    text="Test content",
                    source=CTDTRetrieveSourceMeta(
                        update_cycle_id="15",
                        program_code="7480201",
                    ),
                ),
            ],
            scoped_document_count=3,
            latency_ms=150,
        )
        assert resp.contexts[0].ai_document_id == 42
        assert resp.scoped_document_count == 3

    def test_request_requires_update_cycle_id(self):
        from app.api.v1.ctdt import CTDTRetrieveRequest
        with pytest.raises(Exception):  # pydantic ValidationError
            CTDTRetrieveRequest(query="Test")

    def test_request_requires_query(self):
        from app.api.v1.ctdt import CTDTRetrieveRequest
        with pytest.raises(Exception):
            CTDTRetrieveRequest(update_cycle_id="15")


# ── Sample data guard ────────────────────────────────────────────────


class TestNoSampleDataInApp:
    """Ensure no hardcoded sample data (CSE101, MT1, etc.) in app/."""

    def test_no_sample_data_in_retrieval_service(self):
        import inspect
        from app.services import ctdt_retrieval_service
        source = inspect.getsource(ctdt_retrieval_service)
        assert "CSE101" not in source
        assert "MT1" not in source
        assert "Nhập môn CNTT" not in source

    def test_no_sample_data_in_ctdt_api(self):
        import inspect
        from app.api.v1 import ctdt
        source = inspect.getsource(ctdt)
        assert "CSE101" not in source
        assert "MT1" not in source


# ── Backward compatibility ───────────────────────────────────────────


class TestBackwardCompatibility:
    """Ensure R1/R2 endpoints are still importable and schemas intact."""

    def test_ctdt_ingest_endpoint_exists(self):
        from app.api.v1.ctdt import ctdt_ingest_document
        assert callable(ctdt_ingest_document)

    def test_ctdt_query_endpoint_exists(self):
        from app.api.v1.ctdt import ctdt_query
        assert callable(ctdt_query)

    def test_ctdt_review_endpoint_exists(self):
        from app.api.v1.ctdt import ctdt_review
        assert callable(ctdt_review)

    def test_ctdt_document_status_exists(self):
        from app.api.v1.ctdt import ctdt_get_document_status
        assert callable(ctdt_get_document_status)

    def test_ctdt_retrieve_endpoint_exists(self):
        """New R3 endpoint should be importable."""
        from app.api.v1.ctdt import ctdt_retrieve
        assert callable(ctdt_retrieve)

    def test_extract_text_still_works(self):
        """R2 extract_text backward compat."""
        from app.services.document_extract import extract_text
        result = extract_text("t.txt", "text/plain", b"hello")
        assert isinstance(result, str)


# ── R3.1: Scoped retrieval — allowed_document_ids ────────────────────


class TestScopedRetrieval:
    """R3.1: Verify scoped_doc_ids is passed to QueryService.query() via
    allowed_document_ids to restrict retrieval at vector/BM25 layer."""

    def _make_mock_doc(self, doc_id, update_cycle_id, document_role):
        doc = MagicMock()
        doc.id = doc_id
        doc.tenant_id = "t1"
        doc.source = "cn_ctdt"
        doc.status = "ready"
        doc.title = f"doc_{doc_id}.docx"
        doc.meta = {
            "ctdt": {
                "update_cycle_id": str(update_cycle_id),
                "document_role": document_role,
                "program_code": "7480201",
                "program_id": None,
                "external_file_id": f"file_{doc_id}",
            }
        }
        return doc

    @pytest.mark.asyncio
    async def test_query_svc_receives_allowed_document_ids(self):
        """QueryService.query() must be called with allowed_document_ids=scoped_doc_ids."""
        from app.services.ctdt_retrieval_service import ctdt_retrieve, CTDTTaskType

        docs = [
            self._make_mock_doc(1, "15", "current_curriculum"),
            self._make_mock_doc(2, "15", "survey_evidence"),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        # Mock QueryService
        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        result = await ctdt_retrieve(
            mock_db,
            tenant_id="t1",
            user_id=1,
            query="Test query",
            update_cycle_id="15",
            task_type=CTDTTaskType.GENERAL_QUERY,
            query_svc=mock_query_svc,
        )

        # Assert query_svc.query was called
        mock_query_svc.query.assert_called_once()

        # Assert allowed_document_ids was passed with scoped doc IDs
        call_kwargs = mock_query_svc.query.call_args.kwargs
        assert "allowed_document_ids" in call_kwargs
        assert call_kwargs["allowed_document_ids"] == {1, 2}

    @pytest.mark.asyncio
    async def test_scoped_ids_match_update_cycle(self):
        """Only cycle 15 doc IDs should be passed to QueryService, not cycle 16."""
        from app.services.ctdt_retrieval_service import ctdt_retrieve, CTDTTaskType

        docs = [
            self._make_mock_doc(1, "15", "current_curriculum"),
            self._make_mock_doc(2, "16", "current_curriculum"),  # wrong cycle
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        await ctdt_retrieve(
            mock_db,
            tenant_id="t1",
            user_id=1,
            query="Test",
            update_cycle_id="15",
            task_type=CTDTTaskType.GENERAL_QUERY,
            query_svc=mock_query_svc,
        )

        call_kwargs = mock_query_svc.query.call_args.kwargs
        # Only doc 1 (cycle 15) should be in allowed_document_ids
        assert call_kwargs["allowed_document_ids"] == {1}
        assert 2 not in call_kwargs["allowed_document_ids"]

    @pytest.mark.asyncio
    async def test_post_filter_still_guards_leakage(self):
        """Even with allowed_document_ids, post-filter should strip leaked results."""
        from app.services.ctdt_retrieval_service import ctdt_retrieve, CTDTTaskType
        from app.services.retrieval.types import QueryResult

        docs = [self._make_mock_doc(1, "15", "current_curriculum")]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        # Simulate QueryService returning a result from doc 99 (not in scope)
        leaked_result = QueryResult(
            chunk_id=9900005, document_id=99, score=0.9,
            snippet="Leaked!", highlights=(),
        )
        in_scope_result = QueryResult(
            chunk_id=100005, document_id=1, score=0.8,
            snippet="In scope", highlights=(),
        )

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[leaked_result, in_scope_result])

        result = await ctdt_retrieve(
            mock_db,
            tenant_id="t1",
            user_id=1,
            query="Test",
            update_cycle_id="15",
            task_type=CTDTTaskType.GENERAL_QUERY,
            query_svc=mock_query_svc,
        )

        # Only doc 1 should survive, doc 99 filtered out
        assert len(result.contexts) == 1
        assert result.contexts[0].ai_document_id == 1

    @pytest.mark.asyncio
    async def test_role_scoping_flows_to_query_svc(self):
        """Role filter should restrict the doc IDs passed to QueryService."""
        from app.services.ctdt_retrieval_service import ctdt_retrieve, CTDTTaskType

        docs = [
            self._make_mock_doc(1, "15", "current_curriculum"),
            self._make_mock_doc(2, "15", "survey_evidence"),
            self._make_mock_doc(3, "15", "meeting_report"),
        ]

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = docs

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)

        mock_query_svc = AsyncMock()
        mock_query_svc.query = AsyncMock(return_value=[])

        # Explicit roles: only current_curriculum
        await ctdt_retrieve(
            mock_db,
            tenant_id="t1",
            user_id=1,
            query="Test",
            update_cycle_id="15",
            document_roles=["current_curriculum"],
            task_type=CTDTTaskType.GENERAL_QUERY,
            query_svc=mock_query_svc,
        )

        call_kwargs = mock_query_svc.query.call_args.kwargs
        # Only doc 1 (current_curriculum) should be in scope
        assert call_kwargs["allowed_document_ids"] == {1}


class TestQueryServiceAllowedDocumentIds:
    """Test QueryService.query() backward compat with allowed_document_ids."""

    def test_query_signature_accepts_allowed_document_ids(self):
        """QueryService.query() should accept allowed_document_ids parameter."""
        import inspect
        from app.services.retrieval.query_service import QueryService
        sig = inspect.signature(QueryService.query)
        assert "allowed_document_ids" in sig.parameters
        param = sig.parameters["allowed_document_ids"]
        assert param.default is None  # backward-compatible default

