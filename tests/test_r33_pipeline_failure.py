"""
Tests for R3.3 — Pipeline Failure Semantics + Legacy Stats Backfill.

Tests cover:
1. Embedding/index failure propagation from _embed_and_index()
2. VECTOR_INDEX=null treated as valid (not error)
3. Legacy READY docs missing pipeline stats → force reprocess
4. Legacy READY docs with full stats → no reprocess
5. Stats gap detection helper
6. pipeline_stats_missing marker for non-force-reprocess path
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass

from app.services.document_service import DocumentPipelineStats, DocumentService
from app.services.document_lifecycle import READY, ERROR, CHUNKED, UPLOADED


# ── Helpers ──────────────────────────────────────────────────────────


def _make_mock_doc(
    *,
    doc_id: int = 1,
    status: str = READY,
    content_text: str = "Some real content here.",
    checksum: str = "abc123",
    meta: dict | None = None,
    tenant_id: str = "t1",
    source: str = "cn_ctdt",
    external_id: str = "ext_1",
    title: str = "test.pdf",
):
    doc = MagicMock()
    doc.id = doc_id
    doc.tenant_id = tenant_id
    doc.source = source
    doc.external_id = external_id
    doc.title = title
    doc.status = status
    doc.checksum = checksum
    doc.version_id = checksum
    doc.content_text = content_text
    doc.content_raw = content_text
    doc.representation_type = "original"
    doc.parent_document_id = None
    doc.meta = meta or {}
    doc.created_at = None
    doc.updated_at = None
    return doc


# ══════════════════════════════════════════════════════════════════════
# Vấn đề 1: Pipeline Failure Semantics
# ══════════════════════════════════════════════════════════════════════


class TestEmbedIndexFailurePropagation:
    """_embed_and_index() must re-raise after setting status=ERROR."""

    @pytest.mark.asyncio
    async def test_embed_and_index_raises_on_failure(self):
        """When vector_index.upsert raises, _embed_and_index should set
        doc.status=ERROR then re-raise the original exception."""
        service = DocumentService()

        # Mock vector_index that raises on upsert
        mock_vi = MagicMock()
        mock_vi.upsert = AsyncMock(side_effect=RuntimeError("embedding failed"))
        service._vector_index = mock_vi

        # Mock embedding provider
        mock_embed = MagicMock()
        mock_embed.embed = AsyncMock(return_value=[[0.1, 0.2]])
        service._embedding_provider = mock_embed

        doc = _make_mock_doc(status=CHUNKED)
        mock_db = AsyncMock()

        # Create a minimal chunk
        chunk = MagicMock()
        chunk.text = "test text"
        chunk.token_count = 5

        with pytest.raises(RuntimeError, match="embedding failed"):
            await service._embed_and_index(
                mock_db,
                doc=doc,
                chunks=[chunk],
                old_version=None,
            )

        # doc.status should be ERROR after the exception
        assert doc.status == ERROR

    @pytest.mark.asyncio
    async def test_upsert_does_not_return_success_on_index_failure(self):
        """DocumentService.upsert() must propagate embedding/index exceptions,
        not silently return a success tuple."""
        service = DocumentService()

        # Mock repo: no existing doc → create path
        mock_repo = MagicMock()
        mock_repo.get_by_key = AsyncMock(return_value=None)
        mock_repo.add = MagicMock()
        service.repo = mock_repo

        # Mock vector_index that raises
        mock_vi = MagicMock()
        mock_vi.upsert = AsyncMock(side_effect=RuntimeError("index failed"))
        service._vector_index = mock_vi

        # Mock embedding provider
        mock_embed = MagicMock()
        mock_embed.embed = AsyncMock(return_value=[[0.1, 0.2]])
        service._embedding_provider = mock_embed

        mock_db = AsyncMock()

        with patch.object(DocumentService, '_process_content', return_value=("cleaned", [MagicMock(text="t", token_count=5)])):
            with patch.object(DocumentService, '_configured_vector_index_name', return_value="pgvector"):
                with patch.object(DocumentService, '_configured_embedding_provider_name', return_value="local"):
                    with pytest.raises(RuntimeError, match="index failed"):
                        await service.upsert(
                            db=mock_db,
                            tenant_id="t1",
                            source="test",
                            external_id="ext1",
                            content="some content",
                            title="test.pdf",
                            metadata={},
                        )

    @pytest.mark.asyncio
    async def test_ctdt_ingest_catches_index_failure(self):
        """CTDTIngestService should catch index failure and raise IngestPipelineError."""
        from app.services.ctdt_ingest_service import (
            IngestPipelineError,
            ingest_from_url,
        )

        mock_db = AsyncMock()

        with patch("app.services.ctdt_ingest_service.validate_file_url"):
            with patch("app.services.ctdt_ingest_service.validate_file_url_host"):
                with patch("app.services.ctdt_ingest_service.validate_mime_type"):
                    with patch("app.services.ctdt_ingest_service.download_remote_file", new_callable=AsyncMock, return_value=b"file data"):
                        with patch("app.services.ctdt_ingest_service.extract_text_with_metadata", return_value=("extracted text", {})):
                            # Make DocumentService.upsert raise (simulating index failure)
                            with patch("app.services.ctdt_ingest_service.DocumentService") as MockDocSvc:
                                mock_svc_instance = MagicMock()
                                mock_svc_instance.upsert = AsyncMock(
                                    side_effect=RuntimeError("vector index connection refused")
                                )
                                MockDocSvc.return_value = mock_svc_instance

                                with pytest.raises(IngestPipelineError) as exc_info:
                                    await ingest_from_url(
                                        mock_db,
                                        tenant_id="t1",
                                        external_file_id="file_1",
                                        file_url="https://fileserver.local/file.pdf",
                                        filename="test.pdf",
                                        mime_type="application/pdf",
                                        checksum=None,
                                        update_cycle_id="15",
                                        program_id=None,
                                        program_code="7480201",
                                        program_name="CNTT",
                                        document_role="current_curriculum",
                                        uploaded_by="user1",
                                    )

                                assert exc_info.value.error.code == "index_failed"
                                assert exc_info.value.error.retryable is True


class TestNullVectorIndexNotError:
    """VECTOR_INDEX=null → indexed=False is valid, not an error."""

    @pytest.mark.asyncio
    async def test_null_vector_index_returns_success(self):
        """With NullIndex, _embed_and_index should return (0, 0, False) successfully."""
        from app.services.vector_index import NullIndex

        service = DocumentService()
        service._vector_index = NullIndex()

        doc = _make_mock_doc(status=CHUNKED)
        mock_db = AsyncMock()

        result = await service._embed_and_index(
            mock_db,
            doc=doc,
            chunks=[],
            old_version=None,
        )

        assert result == (0, 0, False)
        assert doc.status == READY  # transitioned to READY, not ERROR

    def test_null_vector_stats_indexed_false_is_valid(self):
        """_stats_from_existing_doc with null vector_index should have indexed=False."""
        doc = _make_mock_doc(meta={
            "pipeline": {
                "vector_index": "null",
                "chunk_count": 10,
                "cleaned_text_length": 500,
                "embedding_count": 0,
                "indexed_count": 0,
                "indexed": False,
                "embedding_provider": "local",
            },
        })

        stats = DocumentService._stats_from_existing_doc(doc)
        assert stats.indexed is False
        assert stats.embedding_count == 0
        assert stats.indexed_count == 0
        assert stats.chunk_count == 10
        # This is NOT an error — just no-vector mode

    def test_null_vector_ingest_stats_no_error(self):
        """_persist_ingest_stats with null vector should set indexed=False cleanly."""
        from app.services.ctdt_ingest_service import _persist_ingest_stats
        from app.services.document_service import DocumentPipelineStats

        doc = _make_mock_doc()

        stats = DocumentPipelineStats(
            cleaned_text_length=500,
            chunk_count=10,
            embedding_count=0,
            indexed_count=0,
            indexed=False,
            vector_index="null",
            embedding_provider="local",
        )

        _persist_ingest_stats(doc, 500, 10, "ready", pipeline_stats=stats)

        assert doc.meta["ctdt"]["indexed"] is False
        assert doc.meta["ctdt"]["vector_index"] == "null"
        assert doc.meta["pipeline"]["indexed"] is False
        assert doc.meta["ctdt"]["ingest_status"] == "ready"


# ══════════════════════════════════════════════════════════════════════
# Vấn đề 2: Legacy Stats Backfill
# ══════════════════════════════════════════════════════════════════════


class TestStatsGapDetection:
    """Unit tests for DocumentService._has_pipeline_stats_gap()."""

    def test_no_pipeline_meta_is_gap(self):
        """Doc with no pipeline meta and no ctdt.chunk_count → gap."""
        doc = _make_mock_doc(meta={"ctdt": {"external_file_id": "f1"}})
        assert DocumentService._has_pipeline_stats_gap(doc) is True

    def test_no_meta_at_all_is_gap(self):
        doc = _make_mock_doc(meta=None)
        assert DocumentService._has_pipeline_stats_gap(doc) is True

    def test_empty_meta_is_gap(self):
        doc = _make_mock_doc(meta={})
        assert DocumentService._has_pipeline_stats_gap(doc) is True

    def test_chunk_count_zero_with_content_is_gap(self):
        """chunk_count=0 but content_text has data → gap."""
        doc = _make_mock_doc(
            content_text="Real document content here",
            meta={"pipeline": {"chunk_count": 0}},
        )
        assert DocumentService._has_pipeline_stats_gap(doc) is True

    def test_chunk_count_positive_no_gap(self):
        """chunk_count > 0 → not a gap."""
        doc = _make_mock_doc(
            content_text="Some content",
            meta={"pipeline": {"chunk_count": 10, "cleaned_text_length": 500}},
        )
        assert DocumentService._has_pipeline_stats_gap(doc) is False

    def test_ctdt_chunk_count_positive_no_gap(self):
        """chunk_count in ctdt meta → not a gap."""
        doc = _make_mock_doc(
            content_text="Some content",
            meta={"ctdt": {"chunk_count": 5}},
        )
        assert DocumentService._has_pipeline_stats_gap(doc) is False

    def test_no_content_text_zero_chunks_no_gap(self):
        """Empty content_text + chunk_count=0 → not a gap (legitimately empty)."""
        doc = _make_mock_doc(
            content_text="",
            meta={"pipeline": {"chunk_count": 0}},
        )
        assert DocumentService._has_pipeline_stats_gap(doc) is False

    def test_none_content_text_no_gap(self):
        """None content_text + no chunks → not a gap (no content to chunk)."""
        doc = _make_mock_doc(content_text=None, meta={"pipeline": {"chunk_count": 0}})
        # content is None/empty → chunk_count=0 is legitimate
        assert DocumentService._has_pipeline_stats_gap(doc) is False


class TestForceReprocessLegacyStats:
    """Test force_reprocess_if_pipeline_stats_missing behaviour."""

    def test_should_reprocess_with_flag_and_gap(self):
        """READY doc with stats gap + force flag → should reprocess."""
        doc = _make_mock_doc(
            status=READY,
            checksum="abc123",
            content_text="Real content",
            meta={"ctdt": {"external_file_id": "f1"}},  # no pipeline meta
        )
        metadata = {
            "system": {"force_reprocess_if_pipeline_stats_missing": True},
        }
        result = DocumentService._should_reprocess_content(
            existing_doc=doc,
            new_checksum="abc123",  # same checksum
            metadata=metadata,
        )
        assert result is True

    def test_should_not_reprocess_without_flag(self):
        """READY doc with stats gap but NO force flag → no reprocess."""
        doc = _make_mock_doc(
            status=READY,
            checksum="abc123",
            content_text="Real content",
            meta={"ctdt": {"external_file_id": "f1"}},
        )
        metadata = {"system": {}}
        result = DocumentService._should_reprocess_content(
            existing_doc=doc,
            new_checksum="abc123",
            metadata=metadata,
        )
        assert result is False

    def test_should_not_reprocess_with_full_stats(self):
        """READY doc with full pipeline stats + force flag → no reprocess."""
        doc = _make_mock_doc(
            status=READY,
            checksum="abc123",
            content_text="Real content",
            meta={
                "pipeline": {
                    "chunk_count": 10,
                    "cleaned_text_length": 500,
                    "embedding_count": 10,
                    "indexed_count": 10,
                    "indexed": True,
                    "vector_index": "pgvector",
                    "embedding_provider": "local",
                },
            },
        )
        metadata = {
            "system": {"force_reprocess_if_pipeline_stats_missing": True},
        }
        result = DocumentService._should_reprocess_content(
            existing_doc=doc,
            new_checksum="abc123",
            metadata=metadata,
        )
        assert result is False

    def test_non_ctdt_doc_not_affected(self):
        """Non-CTDT doc (no force flag in metadata) → no reprocess even with gap."""
        doc = _make_mock_doc(
            status=READY,
            checksum="abc123",
            content_text="Real content",
            meta={},
        )
        # Generic metadata without the force flag
        metadata = {"system": {"pipeline_mode": "legacy"}}
        result = DocumentService._should_reprocess_content(
            existing_doc=doc,
            new_checksum="abc123",
            metadata=metadata,
        )
        assert result is False

    def test_checksum_changed_always_reprocess(self):
        """Different checksum → always reprocess, regardless of stats."""
        doc = _make_mock_doc(
            status=READY,
            checksum="old_hash",
            content_text="Real content",
            meta={
                "pipeline": {"chunk_count": 10},
            },
        )
        result = DocumentService._should_reprocess_content(
            existing_doc=doc,
            new_checksum="new_hash",
        )
        assert result is True

    def test_non_ready_always_reprocess(self):
        """Non-READY doc → always reprocess."""
        doc = _make_mock_doc(
            status=ERROR,
            checksum="abc123",
        )
        result = DocumentService._should_reprocess_content(
            existing_doc=doc,
            new_checksum="abc123",
        )
        assert result is True


class TestPipelineStatsMissingMarker:
    """When a READY doc has a stats gap but is NOT force-reprocessed,
    the noop/fast-update path should mark pipeline_stats_missing=True."""

    @pytest.mark.asyncio
    async def test_noop_path_marks_stats_missing(self):
        """NOOP path should set pipeline_stats_missing=True when stats gap exists."""
        import hashlib
        content = "Real content here"
        real_checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()

        doc = _make_mock_doc(
            status=READY,
            checksum=real_checksum,  # same checksum as content → NOOP path
            content_text=content,
            meta={"ctdt": {"external_file_id": "f1"}},  # no pipeline meta
        )

        service = DocumentService()
        mock_repo = MagicMock()
        mock_repo.get_by_key = AsyncMock(return_value=doc)
        service.repo = mock_repo

        mock_db = AsyncMock()

        with patch.object(DocumentService, '_configured_vector_index_name', return_value="null"):
            with patch.object(DocumentService, '_configured_embedding_provider_name', return_value="local"):
                result_doc, action, changed, stats = await service.upsert(
                    db=mock_db,
                    tenant_id="t1",
                    source="cn_ctdt",
                    external_id="ext_1",
                    content=content,
                    title="test.pdf",
                    metadata={"system": {}},  # no force flag
                )

        # Should mark pipeline_stats_missing
        assert result_doc.meta.get("pipeline", {}).get("pipeline_stats_missing") is True


class TestForceReprocessIntegration:
    """Integration: CTDT metadata includes force flag, triggering reprocess for legacy docs."""

    def test_ctdt_metadata_includes_force_flag(self):
        """build_ctdt_metadata() should include force_reprocess_if_pipeline_stats_missing."""
        from app.services.ctdt_ingest_service import build_ctdt_metadata

        meta = build_ctdt_metadata(
            external_file_id="f1",
            update_cycle_id="15",
            program_id=None,
            program_code="7480201",
            program_name="CNTT",
            document_role="current_curriculum",
            uploaded_by="user1",
            checksum=None,
            filename="test.pdf",
            mime_type="application/pdf",
            file_size_bytes=1024,
            ingest_mode="legacy",
        )

        assert meta["system"]["force_reprocess_if_pipeline_stats_missing"] is True


# ══════════════════════════════════════════════════════════════════════
# Vấn đề 3: Documentation
# ══════════════════════════════════════════════════════════════════════


class TestCtdtDocumentation:
    """Verify ctdt.py module docstring is updated."""

    def test_docstring_mentions_processing_engine(self):
        import app.api.v1.ctdt as ctdt_module
        docstring = ctdt_module.__doc__
        assert "Processing Engine" in docstring or "AI Processing Engine" in docstring

    def test_docstring_mentions_backward_compatible(self):
        import app.api.v1.ctdt as ctdt_module
        docstring = ctdt_module.__doc__
        assert "backward" in docstring.lower() or "Legacy" in docstring

    def test_docstring_no_longer_says_moodle_oriented(self):
        """Module docstring should not describe itself as Moodle-oriented."""
        import app.api.v1.ctdt as ctdt_module
        docstring = ctdt_module.__doc__
        assert "Tích hợp với Moodle LMS" not in docstring

    def test_docstring_mentions_future_analyze(self):
        import app.api.v1.ctdt as ctdt_module
        docstring = ctdt_module.__doc__
        assert "analyze" in docstring.lower()


# ══════════════════════════════════════════════════════════════════════
# Existing behaviour preservation
# ══════════════════════════════════════════════════════════════════════


class TestExistingBehaviourPreserved:
    """Regression checks for unmodified behaviours."""

    def test_stats_from_existing_doc_with_full_pipeline(self):
        """Existing doc with full pipeline stats returns them correctly."""
        doc = _make_mock_doc(meta={
            "pipeline": {
                "cleaned_text_length": 1000,
                "chunk_count": 20,
                "embedding_count": 20,
                "indexed_count": 20,
                "indexed": True,
                "vector_index": "pgvector",
                "embedding_provider": "openai",
            },
        })
        stats = DocumentService._stats_from_existing_doc(doc)
        assert stats.cleaned_text_length == 1000
        assert stats.chunk_count == 20
        assert stats.embedding_count == 20
        assert stats.indexed is True
        assert stats.vector_index == "pgvector"
        assert stats.embedding_provider == "openai"

    def test_document_pipeline_stats_to_dict(self):
        stats = DocumentPipelineStats(
            cleaned_text_length=500,
            chunk_count=10,
            embedding_count=10,
            indexed_count=10,
            indexed=True,
            vector_index="pgvector",
            embedding_provider="local",
        )
        d = stats.to_dict()
        assert d["chunk_count"] == 10
        assert d["indexed"] is True
        assert d["vector_index"] == "pgvector"
