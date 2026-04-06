"""
Phase 6 — Document-level ACL integration tests.

Covers the critical metadata-first bypass fix + role/permission wiring:
  1. Restricted docs are filtered out of metadata-first results
  2. Only ACL-passing docs can trigger early-return
  3. Mixed tenant-wide + restricted docs are correctly handled
  4. Malformed access_scope is denied (not tenant-wide)
  5. Role/permission context is wired through to evaluator
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.metadata_search import MetadataCandidate
from app.services.retrieval.document_access_evaluator import UserAccessContext
from app.services.retrieval.query_service import QueryService


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_candidate(
    doc_id: int,
    score: float = 0.80,
    access_scope: dict | None = None,
) -> MetadataCandidate:
    """Build a MetadataCandidate with optional source_platform access_scope."""
    meta = {}
    if access_scope is not None:
        meta["source_platform"] = {"access_scope": access_scope}

    return MetadataCandidate(
        document_id=doc_id,
        title=f"Doc {doc_id}",
        source="test",
        external_id=f"ext-{doc_id}",
        representation_type="original",
        metadata=meta,
        content_text=f"Content of doc {doc_id}",
        score=score,
        reasons=["test"],
    )


def _make_query_service(*, metadata_first_candidates=None):
    """Build a QueryService with mocked dependencies."""
    mock_access = AsyncMock()
    mock_access.allowed_documents = AsyncMock(return_value={1, 2, 3, 4, 5})

    mock_embedding = AsyncMock()
    mock_embedding.embed = AsyncMock(return_value=[[0.1, 0.2]])

    mock_vector = AsyncMock()
    mock_vector.search = AsyncMock(return_value=[])

    mock_bm25 = AsyncMock()
    mock_bm25.search = AsyncMock(return_value=[])

    mock_reranker = AsyncMock()
    mock_reranker.rerank = AsyncMock(return_value=[])

    mock_response_builder = MagicMock()
    mock_response_builder.build = MagicMock(return_value=[])

    mock_metadata_first = None
    if metadata_first_candidates is not None:
        mock_metadata_first = MagicMock()
        mock_metadata_first.retrieve = AsyncMock(
            return_value=metadata_first_candidates,
        )
        # Always good enough — the test validates ACL filtering intervenes
        mock_metadata_first.is_good_enough = MagicMock(return_value=True)

    svc = QueryService(
        vector_retriever=mock_vector,
        bm25_retriever=mock_bm25,
        embedding_provider=mock_embedding,
        access_policy=mock_access,
        reranker=mock_reranker,
        response_builder=mock_response_builder,
        metadata_first_retrieval_service=mock_metadata_first,
    )
    return svc


def _user_ctx(user_id, role_codes=None, permissions=None):
    """Build a UserAccessContext for tests."""
    return UserAccessContext.from_query_caller(
        user_id=user_id,
        role_codes=role_codes,
        permissions=permissions,
    )


# =====================================================================
# 1. METADATA-FIRST ACL FILTERING (UNIT)
# =====================================================================


class TestMetadataFirstAclFiltering:
    """Test that _apply_acl_to_metadata_candidates correctly filters."""

    def test_tenant_wide_docs_pass_through(self):
        """Docs without access_scope pass through unchanged."""
        svc = _make_query_service()
        candidates = [
            _make_candidate(1),
            _make_candidate(2),
        ]
        result = svc._apply_acl_to_metadata_candidates(
            candidates, acl_user_ctx=_user_ctx(10),
        )
        assert len(result) == 2

    def test_restricted_doc_denied_for_wrong_user(self):
        """Restricted doc not accessible to the querying user."""
        svc = _make_query_service()
        candidates = [
            _make_candidate(1),  # tenant-wide
            _make_candidate(2, access_scope={
                "visibility": "restricted",
                "allow_user_ids": ["99"],  # not user 10
            }),
        ]
        result = svc._apply_acl_to_metadata_candidates(
            candidates, acl_user_ctx=_user_ctx(10),
        )
        assert len(result) == 1
        assert result[0].document_id == 1

    def test_restricted_doc_allowed_for_matching_user(self):
        """Restricted doc accessible to the querying user."""
        svc = _make_query_service()
        candidates = [
            _make_candidate(1, access_scope={
                "visibility": "restricted",
                "allow_user_ids": ["10"],
            }),
        ]
        result = svc._apply_acl_to_metadata_candidates(
            candidates, acl_user_ctx=_user_ctx(10),
        )
        assert len(result) == 1
        assert result[0].document_id == 1

    def test_all_restricted_denied_falls_through(self):
        """When all candidates are restricted and denied, empty result."""
        svc = _make_query_service()
        candidates = [
            _make_candidate(1, access_scope={
                "visibility": "restricted",
                "allow_user_ids": ["99"],
            }),
            _make_candidate(2, access_scope={
                "visibility": "restricted",
                "allow_user_ids": ["88"],
            }),
        ]
        result = svc._apply_acl_to_metadata_candidates(
            candidates, acl_user_ctx=_user_ctx(10),
        )
        assert len(result) == 0

    def test_empty_candidates_returns_empty(self):
        svc = _make_query_service()
        result = svc._apply_acl_to_metadata_candidates(
            [], acl_user_ctx=_user_ctx(10),
        )
        assert result == []

    def test_malformed_scope_denied(self):
        """Malformed access_scope (wrong type) is denied, not tenant-wide."""
        svc = _make_query_service()
        candidates = [
            _make_candidate(1),  # tenant-wide OK
            _make_candidate(2, access_scope="bad_string"),
        ]
        # access_scope is a string → malformed → denied
        # But _make_candidate only puts it in meta if not None, and it's
        # a string. Let's construct manually.
        bad_candidate = MetadataCandidate(
            document_id=2,
            title="Doc 2",
            source="test",
            external_id="ext-2",
            representation_type="original",
            metadata={"source_platform": {"access_scope": "bad_string"}},
            content_text="Content",
            score=0.80,
            reasons=["test"],
        )
        result = svc._apply_acl_to_metadata_candidates(
            [_make_candidate(1), bad_candidate],
            acl_user_ctx=_user_ctx(10),
        )
        assert len(result) == 1
        assert result[0].document_id == 1

    def test_role_based_access_works_with_context(self):
        """Restricted doc with role_codes accessed by user with matching role."""
        svc = _make_query_service()
        candidates = [
            _make_candidate(1, access_scope={
                "visibility": "restricted",
                "role_codes": ["editor"],
            }),
        ]
        result = svc._apply_acl_to_metadata_candidates(
            candidates,
            acl_user_ctx=_user_ctx(10, role_codes=["editor"]),
        )
        assert len(result) == 1

    def test_role_based_access_denied_without_role(self):
        """Restricted doc with role_codes denied when user lacks role."""
        svc = _make_query_service()
        candidates = [
            _make_candidate(1, access_scope={
                "visibility": "restricted",
                "role_codes": ["admin"],
            }),
        ]
        result = svc._apply_acl_to_metadata_candidates(
            candidates,
            acl_user_ctx=_user_ctx(10, role_codes=["viewer"]),
        )
        assert len(result) == 0

    def test_permission_based_access_works_with_context(self):
        """Restricted doc with permission_keys accessed by user with all perms."""
        svc = _make_query_service()
        candidates = [
            _make_candidate(1, access_scope={
                "visibility": "restricted",
                "permission_keys": ["doc.read", "doc.export"],
            }),
        ]
        result = svc._apply_acl_to_metadata_candidates(
            candidates,
            acl_user_ctx=_user_ctx(
                10, permissions=["doc.read", "doc.export", "doc.write"],
            ),
        )
        assert len(result) == 1

    def test_permission_based_access_denied_partial_match(self):
        """Restricted doc denied when user has only partial permissions."""
        svc = _make_query_service()
        candidates = [
            _make_candidate(1, access_scope={
                "visibility": "restricted",
                "permission_keys": ["doc.read", "doc.admin"],
            }),
        ]
        result = svc._apply_acl_to_metadata_candidates(
            candidates,
            acl_user_ctx=_user_ctx(10, permissions=["doc.read"]),
        )
        assert len(result) == 0


# =====================================================================
# 2. METADATA-FIRST EARLY-RETURN BYPASS FIX (E2E)
# =====================================================================


class TestMetadataFirstBypassFix:
    """Verify that restricted docs cannot trigger early-return
    and bypass the normal access policy.

    This is the critical architectural fix of Phase 6.
    """

    def test_restricted_docs_removed_before_good_enough_check(self):
        """If metadata-first returns restricted-only docs,
        they should be filtered → is_good_enough sees empty list → no early return.
        """
        restricted_candidates = [
            _make_candidate(1, score=0.90, access_scope={
                "visibility": "restricted",
                "allow_user_ids": ["99"],
            }),
            _make_candidate(2, score=0.85, access_scope={
                "visibility": "restricted",
                "allow_user_ids": ["88"],
            }),
            _make_candidate(3, score=0.80, access_scope={
                "visibility": "restricted",
                "allow_user_ids": ["77"],
            }),
        ]

        svc = _make_query_service(
            metadata_first_candidates=restricted_candidates,
        )

        good_enough_calls = []

        def tracking_good_enough(candidates):
            good_enough_calls.append(candidates)
            return len(candidates) >= 3

        svc._metadata_first_retrieval_service.is_good_enough = tracking_good_enough

        # Mock resolve_user_access_context to return user_id-only ctx
        with patch(
            "app.services.retrieval.query_service.resolve_user_access_context",
            new_callable=AsyncMock,
            return_value=_user_ctx(10),
        ):
            results = _run(svc.query(
                tenant_id="t1",
                user_id=10,
                query_text="test query",
            ))

        assert len(good_enough_calls) == 1
        assert len(good_enough_calls[0]) == 0

    def test_mixed_candidates_only_allowed_reach_good_enough(self):
        """Mixed tenant-wide + restricted: only allowed docs
        reach the is_good_enough check.
        """
        mixed_candidates = [
            _make_candidate(1, score=0.90),  # tenant-wide
            _make_candidate(2, score=0.85, access_scope={
                "visibility": "restricted",
                "allow_user_ids": ["99"],
            }),
            _make_candidate(3, score=0.80),  # tenant-wide
        ]

        svc = _make_query_service(
            metadata_first_candidates=mixed_candidates,
        )

        good_enough_calls = []

        def tracking_good_enough(candidates):
            good_enough_calls.append(list(candidates))
            return len(candidates) >= 2 and candidates[0].score >= 0.75

        svc._metadata_first_retrieval_service.is_good_enough = tracking_good_enough

        with patch(
            "app.services.retrieval.query_service.resolve_user_access_context",
            new_callable=AsyncMock,
            return_value=_user_ctx(10),
        ):
            results = _run(svc.query(
                tenant_id="t1",
                user_id=10,
                query_text="test query",
            ))

        assert len(good_enough_calls) == 1
        passed_candidates = good_enough_calls[0]
        assert len(passed_candidates) == 2
        doc_ids = {c.document_id for c in passed_candidates}
        assert doc_ids == {1, 3}


# =====================================================================
# 3. ROLE/PERMISSION CONTEXT WIRING (E2E)
# =====================================================================


class TestRolePermissionContextWiring:
    """Verify that resolve_user_access_context is called and its result
    is used for ACL evaluation in the query pipeline.
    """

    def test_role_restricted_doc_allowed_when_user_has_role(self):
        """E2E: user with 'editor' role sees role-restricted doc."""
        role_candidates = [
            _make_candidate(1, score=0.90, access_scope={
                "visibility": "restricted",
                "role_codes": ["editor"],
            }),
        ]

        svc = _make_query_service(
            metadata_first_candidates=role_candidates,
        )

        good_enough_calls = []

        def tracking_good_enough(candidates):
            good_enough_calls.append(list(candidates))
            return len(candidates) >= 1

        svc._metadata_first_retrieval_service.is_good_enough = tracking_good_enough

        # Simulate resolve_user_access_context returning user with editor role
        with patch(
            "app.services.retrieval.query_service.resolve_user_access_context",
            new_callable=AsyncMock,
            return_value=_user_ctx(10, role_codes=["editor", "viewer"]),
        ):
            results = _run(svc.query(
                tenant_id="t1",
                user_id=10,
                query_text="test query",
            ))

        # Doc should pass ACL (role match) and reach is_good_enough
        assert len(good_enough_calls) == 1
        assert len(good_enough_calls[0]) == 1

    def test_role_restricted_doc_denied_when_user_lacks_role(self):
        """E2E: user without 'admin' role cannot see role-restricted doc."""
        role_candidates = [
            _make_candidate(1, score=0.90, access_scope={
                "visibility": "restricted",
                "role_codes": ["admin"],
            }),
        ]

        svc = _make_query_service(
            metadata_first_candidates=role_candidates,
        )

        good_enough_calls = []

        def tracking_good_enough(candidates):
            good_enough_calls.append(list(candidates))
            return len(candidates) >= 1

        svc._metadata_first_retrieval_service.is_good_enough = tracking_good_enough

        with patch(
            "app.services.retrieval.query_service.resolve_user_access_context",
            new_callable=AsyncMock,
            return_value=_user_ctx(10, role_codes=["viewer"]),
        ):
            results = _run(svc.query(
                tenant_id="t1",
                user_id=10,
                query_text="test query",
            ))

        # Doc should be denied → empty → not good enough
        assert len(good_enough_calls) == 1
        assert len(good_enough_calls[0]) == 0


# =====================================================================
# 4. DEFENSE-IN-DEPTH ACL (NORMAL PATH)
# =====================================================================


class TestDefenseInDepthAcl:
    """Verify _apply_acl_to_reranked does not NameError on denied docs.

    Regression test for the bug where logger.info referenced undefined
    `user_id` variable, causing NameError → silently swallowed by the
    try/except wrapper → fail-open.
    """

    def test_reranked_with_denied_docs_no_exception(self):
        """When denied_count > 0, the logging path must not raise."""
        from app.services.retrieval.types import ScoredChunk

        svc = _make_query_service()

        chunks = [
            ScoredChunk(
                chunk_id=100, document_id=1, version_id="v1",
                chunk_index=0, score=0.9, source="vector",
                snippet="allowed content",
            ),
            ScoredChunk(
                chunk_id=200, document_id=2, version_id="v1",
                chunk_index=0, score=0.8, source="vector",
                snippet="restricted content",
            ),
        ]

        # Doc 1: tenant-wide (no access_scope)
        # Doc 2: restricted, only user 99 allowed
        doc_meta = {
            1: {
                "source": "test", "title": "Doc 1",
                "representation_type": "original",
                "meta": {},
                "created_at": None,
            },
            2: {
                "source": "test", "title": "Doc 2",
                "representation_type": "original",
                "meta": {
                    "source_platform": {
                        "access_scope": {
                            "visibility": "restricted",
                            "allow_user_ids": ["99"],
                        },
                    },
                },
                "created_at": None,
            },
        }

        # Patch on the CLASS because __slots__ prevents instance patching
        with patch.object(
            QueryService, "_load_document_metadata_for_bias",
            new=AsyncMock(return_value=doc_meta),
        ):
            ctx = _user_ctx(10)  # not user 99

            # This must NOT raise NameError
            result = _run(svc._apply_acl_to_reranked(
                chunks,
                tenant_id="t1",
                acl_user_ctx=ctx,
            ))

            assert len(result) == 1
            assert result[0].document_id == 1

    def test_reranked_all_allowed_returns_unchanged(self):
        """When no docs are denied, reranked is returned as-is."""
        from app.services.retrieval.types import ScoredChunk

        svc = _make_query_service()

        chunks = [
            ScoredChunk(
                chunk_id=100, document_id=1, version_id="v1",
                chunk_index=0, score=0.9, source="vector",
                snippet="content",
            ),
        ]

        doc_meta = {
            1: {
                "source": "test", "title": "Doc 1",
                "representation_type": "original",
                "meta": {},
                "created_at": None,
            },
        }

        with patch.object(
            QueryService, "_load_document_metadata_for_bias",
            new=AsyncMock(return_value=doc_meta),
        ):
            ctx = _user_ctx(10)

            result = _run(svc._apply_acl_to_reranked(
                chunks,
                tenant_id="t1",
                acl_user_ctx=ctx,
            ))

            assert len(result) == 1
            assert result[0].document_id == 1


# =====================================================================
# 5. FAIL-CLOSED REGRESSION (PHASE 8.5)
# =====================================================================


class TestPhase85FailClosed:
    """Verify that if _apply_acl_to_reranked raises, query() fail-closes
    (returns empty results) instead of returning unfiltered reranked.

    Regression test for the fail-open bug where the except block only
    logged a warning but let the flow continue with unfiltered reranked.
    """

    def test_acl_exception_produces_empty_results(self):
        """When Phase 8.5 ACL sweep raises, final results should be empty."""
        svc = _make_query_service()

        # Patch _apply_acl_to_reranked to raise, simulating an ACL failure
        with patch.object(
            QueryService, "_apply_acl_to_reranked",
            new=AsyncMock(side_effect=RuntimeError("ACL backend down")),
        ), patch(
            "app.services.retrieval.query_service.resolve_user_access_context",
            new_callable=AsyncMock,
            return_value=_user_ctx(10),
        ):
            results = _run(svc.query(
                tenant_id="t1",
                user_id=10,
                query_text="test query",
            ))

        # The fail-closed behavior clears reranked to []
        # so response builder gets empty input → empty results
        assert results == [] or len(results) == 0
