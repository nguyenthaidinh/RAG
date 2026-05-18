"""
Tests for R4 — Analyze Update Cycle Skeleton.

Covers:
1. Full analysis with retrieval contexts → 7 payload keys present, contexts_count > 0
2. No documents → all 7 keys present but empty, contexts_count = 0
3. ai_document_ids scope validation (pass + reject)
4. Orchestration → all 7 tasks called
5. Source formatting → required fields present
6. Backward compatibility → R1/R2/R3 endpoints still exist
"""
import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ctdt_analysis_service import (
    ALL_PAYLOAD_KEYS,
    ANALYSIS_TASKS,
    ANALYSIS_TASK_SEEDS,
    TASK_PAYLOAD_KEY,
    AnalysisCycleResult,
    AnalysisSkeletonItem,
    AnalysisSource,
    AnalysisSourceSummary,
    AnalysisValidationError,
    _context_to_source,
    _validate_document_scope,
    analyze_update_cycle,
)
from app.services.ctdt_retrieval_service import (
    CTDTRetrievalContext,
    CTDTRetrievalResult,
    CTDTTaskType,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_retrieval_context(
    *,
    doc_id: int = 42,
    chunk_id: int = 4200001,
    score: float = 0.85,
    text: str = "Đây là nội dung trích xuất từ tài liệu.",
    external_file_id: str = "file_1",
    filename: str = "test.pdf",
    document_role: str = "current_curriculum",
    update_cycle_id: str = "15",
    program_code: str = "7480201",
) -> CTDTRetrievalContext:
    return CTDTRetrievalContext(
        ai_document_id=doc_id,
        external_file_id=external_file_id,
        filename=filename,
        document_role=document_role,
        chunk_id=chunk_id,
        chunk_index=chunk_id % 100_000,
        score=score,
        text=text,
        source={
            "update_cycle_id": update_cycle_id,
            "program_code": program_code,
            "program_id": None,
            "section": None,
            "page": None,
        },
    )


def _make_retrieval_result(
    *,
    task_type: CTDTTaskType,
    contexts: list[CTDTRetrievalContext] | None = None,
    update_cycle_id: str = "15",
    program_code: str = "7480201",
) -> CTDTRetrievalResult:
    if contexts is None:
        contexts = [_make_retrieval_context()]
    return CTDTRetrievalResult(
        query=ANALYSIS_TASK_SEEDS.get(task_type, "test"),
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        task_type=task_type.value,
        document_roles_used=[],
        contexts=contexts,
        scoped_document_count=len(set(c.ai_document_id for c in contexts)),
        latency_ms=10,
    )


# ══════════════════════════════════════════════════════════════════════
# Test 1: Full analysis with retrieval contexts
# ══════════════════════════════════════════════════════════════════════


class TestAnalyzeWithContexts:
    """Analyze update cycle with retrieval mock returning contexts."""

    @pytest.mark.asyncio
    async def test_response_has_all_7_payload_keys(self):
        """result_payload must have exactly 7 keys."""

        async def mock_retrieve(db, *, tenant_id, user_id, query, update_cycle_id,
                                program_code, program_id, task_type, document_roles,
                                top_k, query_svc):
            return _make_retrieval_result(task_type=task_type)

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            result = await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="15",
                program_code="7480201",
                program_name="CNTT",
            )

        assert set(result.result_payload.keys()) == set(ALL_PAYLOAD_KEYS)
        for key in ALL_PAYLOAD_KEYS:
            assert key in result.result_payload

    @pytest.mark.asyncio
    async def test_contexts_count_positive(self):
        """source_summary.contexts_count > 0 when retrieval returns results."""

        async def mock_retrieve(db, *, tenant_id, user_id, query, update_cycle_id,
                                program_code, program_id, task_type, document_roles,
                                top_k, query_svc):
            return _make_retrieval_result(task_type=task_type)

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            result = await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="15",
                program_code="7480201",
            )

        assert result.source_summary.contexts_count > 0
        assert len(result.source_summary.documents_used) > 0
        assert result.source_summary.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_skeleton_items_have_needs_generation_status(self):
        """Each skeleton item must have status=needs_generation."""

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            result = await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="15",
            )

        for key, items in result.result_payload.items():
            for item in items:
                assert item.status == "needs_generation"
                assert item.task_type in [t.value for t in ANALYSIS_TASKS]


# ══════════════════════════════════════════════════════════════════════
# Test 2: No documents
# ══════════════════════════════════════════════════════════════════════


class TestAnalyzeNoDocuments:
    """Analyze when no documents match → empty but complete payload."""

    @pytest.mark.asyncio
    async def test_empty_payload_all_keys_present(self):
        """result_payload has all 7 keys but with empty sources."""

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(
                task_type=kwargs["task_type"],
                contexts=[],  # no docs
            )

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            result = await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="99",
            )

        assert set(result.result_payload.keys()) == set(ALL_PAYLOAD_KEYS)
        assert result.source_summary.contexts_count == 0
        assert result.source_summary.documents_used == []

    @pytest.mark.asyncio
    async def test_no_fallback_to_global(self):
        """No retrieval should be attempted outside the update_cycle scope."""

        calls = []

        async def mock_retrieve(db, **kwargs):
            calls.append(kwargs["update_cycle_id"])
            return _make_retrieval_result(
                task_type=kwargs["task_type"],
                contexts=[],
            )

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="99",
            )

        # All calls should be for the same update_cycle
        assert all(c == "99" for c in calls)


# ══════════════════════════════════════════════════════════════════════
# Test 3: ai_document_ids scope validation
# ══════════════════════════════════════════════════════════════════════


class TestDocumentScopeValidation:
    """ai_document_ids must belong to the update_cycle."""

    @pytest.mark.asyncio
    async def test_valid_scope_passes(self):
        """Documents in scope → analysis proceeds normally."""

        # Mock _get_scoped_document_ids to return {42: {...}, 43: {...}}
        scoped = {
            42: {"external_file_id": "f1", "filename": "a.pdf", "document_role": "current_curriculum",
                 "update_cycle_id": "15", "program_code": "7480201"},
            43: {"external_file_id": "f2", "filename": "b.pdf", "document_role": "survey_evidence",
                 "update_cycle_id": "15", "program_code": "7480201"},
        }

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(
                task_type=kwargs["task_type"],
                contexts=[_make_retrieval_context(doc_id=42)],
            )

        with patch("app.services.ctdt_analysis_service._get_scoped_document_ids", new_callable=AsyncMock, return_value=scoped):
            with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
                result = await analyze_update_cycle(
                    AsyncMock(),
                    tenant_id="t1",
                    user_id=1,
                    update_cycle_id="15",
                    ai_document_ids=[42, 43],
                )

        assert result.source_summary.contexts_count > 0

    @pytest.mark.asyncio
    async def test_out_of_scope_raises_error(self):
        """Documents NOT in scope → AnalysisValidationError."""

        # Mock: only doc 42 is in scope
        scoped = {
            42: {"external_file_id": "f1", "filename": "a.pdf", "document_role": "current_curriculum",
                 "update_cycle_id": "15", "program_code": "7480201"},
        }

        with patch("app.services.ctdt_analysis_service._get_scoped_document_ids", new_callable=AsyncMock, return_value=scoped):
            with pytest.raises(AnalysisValidationError) as exc_info:
                await analyze_update_cycle(
                    AsyncMock(),
                    tenant_id="t1",
                    user_id=1,
                    update_cycle_id="15",
                    ai_document_ids=[42, 999],  # 999 not in scope
                )

            assert exc_info.value.code == "invalid_document_scope"
            assert "999" in exc_info.value.message

    @pytest.mark.asyncio
    async def test_ai_doc_ids_filter_sources(self):
        """When ai_document_ids provided, only those docs appear in sources."""

        scoped = {
            42: {"external_file_id": "f1", "filename": "a.pdf", "document_role": "current_curriculum",
                 "update_cycle_id": "15", "program_code": "7480201"},
            43: {"external_file_id": "f2", "filename": "b.pdf", "document_role": "survey_evidence",
                 "update_cycle_id": "15", "program_code": "7480201"},
        }

        async def mock_retrieve(db, **kwargs):
            # Return contexts from both doc 42 and 43
            return _make_retrieval_result(
                task_type=kwargs["task_type"],
                contexts=[
                    _make_retrieval_context(doc_id=42),
                    _make_retrieval_context(doc_id=43),
                    _make_retrieval_context(doc_id=44),  # not in ai_document_ids
                ],
            )

        with patch("app.services.ctdt_analysis_service._get_scoped_document_ids", new_callable=AsyncMock, return_value=scoped):
            with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
                result = await analyze_update_cycle(
                    AsyncMock(),
                    tenant_id="t1",
                    user_id=1,
                    update_cycle_id="15",
                    ai_document_ids=[42, 43],
                )

        # All sources should only be from doc 42 or 43 (not 44)
        all_doc_ids = set()
        for items in result.result_payload.values():
            for item in items:
                for s in item.sources:
                    all_doc_ids.add(s.ai_document_id)

        assert 44 not in all_doc_ids
        assert all_doc_ids <= {42, 43}


# ══════════════════════════════════════════════════════════════════════
# Test 4: Orchestration — all 7 tasks executed
# ══════════════════════════════════════════════════════════════════════


class TestOrchestration:
    """Verify all 7 analysis tasks are called."""

    @pytest.mark.asyncio
    async def test_all_7_tasks_called(self):
        """ctdt_retrieve must be called once per task (7 total)."""

        call_task_types = []

        async def mock_retrieve(db, **kwargs):
            call_task_types.append(kwargs["task_type"])
            return _make_retrieval_result(task_type=kwargs["task_type"], contexts=[])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            result = await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="15",
            )

        assert len(call_task_types) == 7
        expected = set(ANALYSIS_TASKS)
        actual = set(call_task_types)
        assert actual == expected

        assert len(result.source_summary.tasks_executed) == 7

    @pytest.mark.asyncio
    async def test_correct_query_seeds_used(self):
        """Each task should use its designated query seed."""

        used_queries = {}

        async def mock_retrieve(db, **kwargs):
            used_queries[kwargs["task_type"]] = kwargs["query"]
            return _make_retrieval_result(task_type=kwargs["task_type"], contexts=[])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="15",
            )

        for task_type, expected_seed in ANALYSIS_TASK_SEEDS.items():
            assert used_queries[task_type] == expected_seed

    @pytest.mark.asyncio
    async def test_task_failure_does_not_abort_analysis(self):
        """If one task fails, others should still complete."""

        call_count = 0

        async def mock_retrieve(db, **kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["task_type"] == CTDTTaskType.EVIDENCE_ANALYSIS:
                raise RuntimeError("retrieval failed for evidence")
            return _make_retrieval_result(task_type=kwargs["task_type"])

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            result = await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="15",
            )

        # All 7 tasks should be attempted
        assert call_count == 7
        # All keys present
        assert set(result.result_payload.keys()) == set(ALL_PAYLOAD_KEYS)
        # Evidence should have empty sources (failed task)
        evidence_items = result.result_payload["evidence_summary"]
        assert len(evidence_items) == 1
        assert evidence_items[0].sources == []


# ══════════════════════════════════════════════════════════════════════
# Test 5: Source formatting
# ══════════════════════════════════════════════════════════════════════


class TestSourceFormatting:
    """Source objects must have all required fields."""

    def test_context_to_source_has_required_fields(self):
        """_context_to_source must produce AnalysisSource with all fields."""
        ctx = _make_retrieval_context(
            doc_id=42,
            chunk_id=4200005,
            score=0.92,
            text="Nội dung minh chứng khảo sát đầu vào",
            external_file_id="file_1",
            filename="survey.pdf",
            document_role="survey_evidence",
            update_cycle_id="15",
            program_code="7480201",
        )

        source = _context_to_source(ctx)

        assert source.ai_document_id == 42
        assert source.external_file_id == "file_1"
        assert source.filename == "survey.pdf"
        assert source.document_role == "survey_evidence"
        assert source.chunk_id == 4200005
        assert source.chunk_index == 5
        assert source.score == 0.92
        assert "minh chứng" in source.quote
        assert source.update_cycle_id == "15"
        assert source.program_code == "7480201"

    def test_quote_truncated_to_500_chars(self):
        """quote should be truncated to 500 characters."""
        long_text = "A" * 1000
        ctx = _make_retrieval_context(text=long_text)
        source = _context_to_source(ctx)
        assert len(source.quote) == 500

    @pytest.mark.asyncio
    async def test_sources_in_response_have_all_fields(self):
        """Full analysis result sources must have all required fields."""

        async def mock_retrieve(db, **kwargs):
            return _make_retrieval_result(
                task_type=kwargs["task_type"],
                contexts=[_make_retrieval_context()],
            )

        with patch("app.services.ctdt_analysis_service.ctdt_retrieve", side_effect=mock_retrieve):
            result = await analyze_update_cycle(
                AsyncMock(),
                tenant_id="t1",
                user_id=1,
                update_cycle_id="15",
            )

        # Check first source in first non-empty section
        for items in result.result_payload.values():
            for item in items:
                for s in item.sources:
                    assert hasattr(s, "ai_document_id")
                    assert hasattr(s, "external_file_id")
                    assert hasattr(s, "filename")
                    assert hasattr(s, "document_role")
                    assert hasattr(s, "chunk_id")
                    assert hasattr(s, "chunk_index")
                    assert hasattr(s, "score")
                    assert hasattr(s, "quote")
                    assert hasattr(s, "update_cycle_id")
                    assert hasattr(s, "program_code")
                    return  # checked one source, done


# ══════════════════════════════════════════════════════════════════════
# Test 6: Schema / constants correctness
# ══════════════════════════════════════════════════════════════════════


class TestSchemaCorrectness:
    """Verify schema definitions and constants are correct."""

    def test_7_analysis_tasks_defined(self):
        assert len(ANALYSIS_TASKS) == 7

    def test_7_payload_keys_defined(self):
        assert len(ALL_PAYLOAD_KEYS) == 7

    def test_all_tasks_have_seeds(self):
        for task in ANALYSIS_TASKS:
            assert task in ANALYSIS_TASK_SEEDS

    def test_all_tasks_have_payload_keys(self):
        for task in ANALYSIS_TASKS:
            assert task in TASK_PAYLOAD_KEY

    def test_payload_keys_match_spec(self):
        expected = {
            "evidence_summary",
            "evaluation_points",
            "change_proposals",
            "objective_suggestions",
            "outcome_suggestions",
            "course_change_suggestions",
            "matrix_suggestions",
        }
        assert set(ALL_PAYLOAD_KEYS) == expected

    def test_analysis_validation_error(self):
        err = AnalysisValidationError(
            code="invalid_document_scope",
            message="Documents [999] do not belong to update_cycle_id=15",
        )
        assert err.code == "invalid_document_scope"
        assert "999" in str(err)


# ══════════════════════════════════════════════════════════════════════
# Test 7: Backward compatibility
# ══════════════════════════════════════════════════════════════════════


class TestR4BackwardCompatibility:
    """R1/R2/R3 endpoints must still exist."""

    def _get_paths(self):
        from app.api.v1.ctdt import router
        return [r.path for r in router.routes]

    def test_analyze_endpoint_exists(self):
        assert "/api/v1/ctdt/update-cycles/analyze" in self._get_paths()

    def test_ingest_endpoint_still_exists(self):
        assert "/api/v1/ctdt/documents/ingest" in self._get_paths()

    def test_retrieve_endpoint_still_exists(self):
        assert "/api/v1/ctdt/retrieve" in self._get_paths()

    def test_query_endpoint_still_exists(self):
        assert "/api/v1/ctdt/query" in self._get_paths()

    def test_health_endpoint_still_exists(self):
        assert "/api/v1/ctdt/health" in self._get_paths()

    def test_document_status_endpoint_still_exists(self):
        paths = self._get_paths()
        # FastAPI uses {document_id} in path
        assert any("documents" in p and "document_id" in p for p in paths)


# ══════════════════════════════════════════════════════════════════════
# Test 8: Response schema validation
# ══════════════════════════════════════════════════════════════════════


class TestResponseSchema:
    """Verify Pydantic schemas for API response."""

    def test_analyze_request_schema(self):
        from app.api.v1.ctdt import AnalyzeUpdateCycleRequest
        req = AnalyzeUpdateCycleRequest(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
            ai_document_ids=[42, 43],
            analysis_mode="skeleton",
            top_k_per_task=6,
        )
        assert req.update_cycle_id == "15"
        assert req.analysis_mode == "skeleton"
        assert req.ai_document_ids == [42, 43]

    def test_analyze_request_defaults(self):
        from app.api.v1.ctdt import AnalyzeUpdateCycleRequest
        req = AnalyzeUpdateCycleRequest(update_cycle_id="15")
        assert req.analysis_mode == "skeleton"
        assert req.top_k_per_task == 6
        assert req.ai_document_ids is None
        assert req.document_roles is None

    def test_analyze_response_schema(self):
        from app.api.v1.ctdt import (
            AnalyzeUpdateCycleResponse,
            AnalyzeSkeletonItem,
            AnalyzeSourceItem,
            AnalyzeSourceSummary,
        )
        resp = AnalyzeUpdateCycleResponse(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
            analysis_mode="skeleton",
            result_payload={
                "evidence_summary": [
                    AnalyzeSkeletonItem(
                        status="needs_generation",
                        task_type="evidence_analysis",
                        sources=[
                            AnalyzeSourceItem(
                                ai_document_id=42,
                                external_file_id="f1",
                                filename="test.pdf",
                                document_role="survey_evidence",
                                chunk_id=4200001,
                                chunk_index=1,
                                score=0.85,
                                quote="Nội dung minh chứng",
                                update_cycle_id="15",
                                program_code="7480201",
                            )
                        ],
                    )
                ],
            },
            source_summary=AnalyzeSourceSummary(
                contexts_count=1,
                documents_used=[42],
                tasks_executed=["evidence_analysis"],
                latency_ms=50,
            ),
        )
        assert resp.update_cycle_id == "15"
        assert resp.analysis_mode == "skeleton"
        assert len(resp.result_payload["evidence_summary"]) == 1
        assert resp.source_summary.contexts_count == 1
