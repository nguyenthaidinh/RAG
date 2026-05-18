"""Tests for R6.2A Outcome Update Context Pack."""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services.ctdt_objective_context_service import (
    ContextItem,
    ContextPackSourceSummary,
    RoleCoverageItem,
)


# ── Helpers ──────────────────────────────────────────────────────────

# The 7 retrieval role groups in R6.2A
_RETRIEVAL_GROUPS = [
    "current_outcome", "current_curriculum", "direction",
    "legal", "evidence", "comparison", "course_syllabus",
]


def _make_retrieval_result(
    *,
    query: str = "q",
    update_cycle_id: str = "15",
    program_code: str = "7480201",
    document_roles: list[str] | None = None,
    contexts: list | None = None,
    scoped_document_count: int = 1,
):
    """Build a mock CTDTRetrievalResult."""
    from app.services.ctdt_retrieval_service import CTDTRetrievalResult

    if contexts is None:
        contexts = [_make_ctx(document_role=(document_roles or ["current_curriculum"])[0])]

    return CTDTRetrievalResult(
        query=query,
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        task_type="outcome_suggestion",
        document_roles_used=document_roles or ["current_curriculum"],
        contexts=contexts,
        scoped_document_count=scoped_document_count,
        latency_ms=10,
    )


def _make_ctx(
    *,
    ai_document_id: int = 1,
    document_role: str = "current_curriculum",
    filename: str = "ctdt.pdf",
    text: str = "Chuẩn đầu ra PLO hiện hành",
    chunk_index: int = 0,
    score: float = 0.85,
):
    """Build a mock CTDTRetrievalContext."""
    from app.services.ctdt_retrieval_service import CTDTRetrievalContext

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


def _make_empty_result(*, document_roles=None, scoped_document_count=0):
    return _make_retrieval_result(
        document_roles=document_roles,
        contexts=[],
        scoped_document_count=scoped_document_count,
    )


def _make_objective_draft_payload():
    """Simulate a stored objective_update draft payload."""
    return {
        "proposed_objectives": [
            {
                "code": "PO1",
                "is_draft_code": True,
                "proposed_content": "Đào tạo nhân lực CNTT có năng lực AI/ML",
                "objective_type": "general_objective",
            },
        ],
        "missing_information": [],
        "_meta": {"generation_status": "generated", "warnings": []},
    }


# Side-effect for ctdt_retrieve calls — returns result by role
def _side_effect_factory(
    *,
    empty_groups: set[str] | None = None,
    fail_groups: set[str] | None = None,
    doc_no_ctx_groups: set[str] | None = None,
):
    """
    Create a side-effect function for ctdt_retrieve.
    Groups in empty_groups return empty contexts with scoped_document_count=0.
    Groups in doc_no_ctx_groups return empty contexts with scoped_document_count>0.
    Groups in fail_groups raise RuntimeError.
    """
    empty_groups = empty_groups or set()
    fail_groups = fail_groups or set()
    doc_no_ctx_groups = doc_no_ctx_groups or set()

    call_index = [0]

    from app.services.ctdt_outcome_context_service import ROLE_GROUPS

    async def _side_effect(*args, **kwargs):
        roles = kwargs.get("document_roles", [])
        idx = call_index[0]
        call_index[0] += 1

        if idx < len(ROLE_GROUPS):
            key = ROLE_GROUPS[idx]["key"]
        else:
            key = "unknown"

        if key in fail_groups:
            raise RuntimeError(f"retrieval failed for {key}")

        if key in doc_no_ctx_groups:
            return _make_empty_result(
                document_roles=roles,
                scoped_document_count=3,
            )

        if key in empty_groups:
            return _make_empty_result(document_roles=roles)

        return _make_retrieval_result(
            document_roles=roles,
            contexts=[_make_ctx(
                ai_document_id=10 + idx,
                document_role=roles[0] if roles else "current_curriculum",
            )],
        )

    return _side_effect


# ══════════════════════════════════════════════════════════════════════
# 1. Builds context pack with multi-role retrieval
# ══════════════════════════════════════════════════════════════════════


class TestMultiRoleRetrieval:
    @pytest.mark.asyncio
    async def test_builds_context_pack_with_multi_role_retrieval(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=_make_objective_draft_payload(),
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
                program_code="7480201",
            )

        # Should have called retrieval for each of 7 groups
        assert pack.context_pack_type == "outcome_update"
        assert len(pack.source_summary.role_groups_retrieved) == 8  # obj + 7

        # All retrieval groups should have contexts
        for group in _RETRIEVAL_GROUPS:
            assert pack.role_coverage[group].status == "available", (
                f"group {group} should be available"
            )


# ══════════════════════════════════════════════════════════════════════
# 2. Objective_update draft available
# ══════════════════════════════════════════════════════════════════════


class TestObjectiveDraftAvailable:
    @pytest.mark.asyncio
    async def test_objective_draft_available_sets_coverage(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=_make_objective_draft_payload(),
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert pack.role_coverage["objective_update"].status == "available"
        assert pack.objective_update_payload is not None
        assert len(pack.objective_update_payload["proposed_objectives"]) >= 1


# ══════════════════════════════════════════════════════════════════════
# 3. No objective draft → missing_information
# ══════════════════════════════════════════════════════════════════════


class TestObjectiveDraftMissing:
    @pytest.mark.asyncio
    async def test_no_objective_draft_adds_missing_info(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert pack.role_coverage["objective_update"].status == "missing"
        mi_types = [m["type"] for m in pack.missing_information]
        assert "objective_update" in mi_types


# ══════════════════════════════════════════════════════════════════════
# 4. current_outcome contexts populated
# ══════════════════════════════════════════════════════════════════════


class TestCurrentOutcomeContexts:
    @pytest.mark.asyncio
    async def test_current_outcome_contexts_populated(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert len(pack.current_outcome_contexts) >= 1
        assert pack.role_coverage["current_outcome"].context_count >= 1


# ══════════════════════════════════════════════════════════════════════
# 5. direction_contexts populated
# ══════════════════════════════════════════════════════════════════════


class TestDirectionContexts:
    @pytest.mark.asyncio
    async def test_direction_contexts_populated(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert len(pack.direction_contexts) >= 1
        assert pack.role_coverage["direction"].status == "available"


# ══════════════════════════════════════════════════════════════════════
# 6. legal_contexts populated
# ══════════════════════════════════════════════════════════════════════


class TestLegalContexts:
    @pytest.mark.asyncio
    async def test_legal_contexts_populated(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert len(pack.legal_contexts) >= 1
        assert pack.role_coverage["legal"].status == "available"


# ══════════════════════════════════════════════════════════════════════
# 7. course_syllabus_contexts populated
# ══════════════════════════════════════════════════════════════════════


class TestCourseSyllabusContexts:
    @pytest.mark.asyncio
    async def test_course_syllabus_contexts_populated(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert len(pack.course_syllabus_contexts) >= 1
        assert pack.role_coverage["course_syllabus"].status == "available"


# ══════════════════════════════════════════════════════════════════════
# 8. document_available_no_context
# ══════════════════════════════════════════════════════════════════════


class TestDocAvailableNoContext:
    @pytest.mark.asyncio
    async def test_scoped_docs_but_no_context(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(
                doc_no_ctx_groups={"current_outcome"},
            ),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        cov = pack.role_coverage["current_outcome"]
        assert cov.status == "document_available_no_context"
        assert cov.scoped_document_count > 0
        assert cov.context_count == 0

        mi_types = [m["type"] for m in pack.missing_information]
        assert "current_outcomes_context_not_found" in mi_types


# ══════════════════════════════════════════════════════════════════════
# 9. Missing (no documents)
# ══════════════════════════════════════════════════════════════════════


class TestMissingStatus:
    @pytest.mark.asyncio
    async def test_no_docs_returns_missing(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(
                empty_groups={"direction"},
            ),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert pack.role_coverage["direction"].status == "missing"
        mi_types = [m["type"] for m in pack.missing_information]
        assert "direction_decision" in mi_types


# ══════════════════════════════════════════════════════════════════════
# 10. Retrieval exception → status=failed
# ══════════════════════════════════════════════════════════════════════


class TestRetrievalException:
    @pytest.mark.asyncio
    async def test_retrieval_exception_does_not_crash_pack(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(
                fail_groups={"legal"},
            ),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        # legal failed but others succeeded
        assert pack.role_coverage["legal"].status == "failed"
        assert pack.role_coverage["legal"].retrieval_status == "failed"
        assert pack.role_coverage["current_outcome"].status == "available"
        # Pack still returned (fail-open)
        assert pack.source_summary is not None


# ══════════════════════════════════════════════════════════════════════
# 11. documents_used unique and sorted
# ══════════════════════════════════════════════════════════════════════


class TestDocumentsUsedUnique:
    @pytest.mark.asyncio
    async def test_documents_used_unique_sorted(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        docs = pack.source_summary.documents_used
        assert docs == sorted(set(docs))


# ══════════════════════════════════════════════════════════════════════
# 12. Endpoint does not call LLM
# ══════════════════════════════════════════════════════════════════════


class TestEndpointNoLLM:
    @pytest.mark.asyncio
    async def test_endpoint_does_not_call_llm(self):
        from app.api.v1.ctdt import (
            OutcomeContextPackRequest,
            build_outcome_context_pack,
        )
        from app.services.ctdt_outcome_context_service import (
            OutcomeUpdateContextPack,
        )

        pack = OutcomeUpdateContextPack(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
        )
        pack.source_summary = ContextPackSourceSummary(
            total_contexts=0,
            documents_used=[],
            role_groups_retrieved=["objective_update"],
            latency_ms=10,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.build_outcome_update_context_pack",
            return_value=pack,
        ) as mock_build:
            response = await build_outcome_context_pack(
                body=OutcomeContextPackRequest(update_cycle_id="15"),
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        assert response.context_pack_type == "outcome_update"
        mock_build.assert_called_once()


# ══════════════════════════════════════════════════════════════════════
# 13. Endpoint does not commit DB
# ══════════════════════════════════════════════════════════════════════


class TestEndpointNoDBCommit:
    @pytest.mark.asyncio
    async def test_endpoint_no_db_commit(self):
        from app.api.v1.ctdt import (
            OutcomeContextPackRequest,
            build_outcome_context_pack,
        )
        from app.services.ctdt_outcome_context_service import (
            OutcomeUpdateContextPack,
        )

        pack = OutcomeUpdateContextPack(
            update_cycle_id="15",
            program_code="7480201",
            program_name="CNTT",
        )
        pack.source_summary = ContextPackSourceSummary(
            total_contexts=0,
            documents_used=[],
            role_groups_retrieved=["objective_update"],
            latency_ms=10,
        )

        db = AsyncMock()

        with patch(
            "app.services.ctdt_outcome_context_service.build_outcome_update_context_pack",
            return_value=pack,
        ):
            await build_outcome_context_pack(
                body=OutcomeContextPackRequest(update_cycle_id="15"),
                request=AsyncMock(),
                db=db,
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        db.commit.assert_not_called()
        db.add.assert_not_called()


# ══════════════════════════════════════════════════════════════════════
# 14. No Program model imports
# ══════════════════════════════════════════════════════════════════════


class TestNoProgramWrites:
    def test_service_has_no_program_model_imports(self):
        import app.services.ctdt_outcome_context_service as mod
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
# 15. Endpoint error handling
# ══════════════════════════════════════════════════════════════════════


class TestEndpointErrorHandling:
    @pytest.mark.asyncio
    async def test_service_error_returns_500(self):
        from app.api.v1.ctdt import (
            OutcomeContextPackRequest,
            build_outcome_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.build_outcome_update_context_pack",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await build_outcome_context_pack(
                    body=OutcomeContextPackRequest(update_cycle_id="15"),
                    request=AsyncMock(),
                    db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                    query_svc=AsyncMock(),
                )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "outcome_context_pack_error"


# ══════════════════════════════════════════════════════════════════════
# 16. context_pack_type == "outcome_update"
# ══════════════════════════════════════════════════════════════════════


class TestContextPackType:
    @pytest.mark.asyncio
    async def test_context_pack_type_value(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(
                empty_groups=set(_RETRIEVAL_GROUPS),
            ),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        assert pack.context_pack_type == "outcome_update"


# ══════════════════════════════════════════════════════════════════════
# 17. All 8 groups in role_groups_retrieved
# ══════════════════════════════════════════════════════════════════════


class TestAllGroupsRetrieved:
    @pytest.mark.asyncio
    async def test_all_groups_in_role_groups_retrieved(self):
        from app.services.ctdt_outcome_context_service import (
            build_outcome_update_context_pack,
        )

        with patch(
            "app.services.ctdt_outcome_context_service.ctdt_retrieve",
            side_effect=_side_effect_factory(),
        ), patch(
            "app.services.ctdt_outcome_context_service._load_latest_objective_draft",
            return_value=None,
        ):
            pack = await build_outcome_update_context_pack(
                AsyncMock(),
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
            )

        retrieved = pack.source_summary.role_groups_retrieved
        assert "objective_update" in retrieved
        for g in _RETRIEVAL_GROUPS:
            assert g in retrieved, f"{g} should be in role_groups_retrieved"
