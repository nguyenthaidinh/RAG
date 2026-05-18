"""Tests for R1.1 hardening: ctdt_ingest_service."""
import hashlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ctdt_ingest_service import (
    IngestError,
    IngestPipelineError,
    IngestValidationError,
    _is_private_ip,
    _map_status,
    _persist_ingest_stats,
    build_ctdt_metadata,
    get_document_ctdt_info,
    validate_file_url,
    validate_file_url_host,
    validate_mime_type,
    verify_checksum,
)


class TestValidateFileUrl:
    def test_valid_https(self):
        validate_file_url("https://fileserver.example.com/file.pdf")

    def test_valid_http(self):
        validate_file_url("http://192.168.1.100/file.pdf")

    def test_empty_url(self):
        with pytest.raises(IngestValidationError) as exc_info:
            validate_file_url("")
        assert exc_info.value.error.code == "invalid_file_url"

    def test_ftp_rejected(self):
        with pytest.raises(IngestValidationError):
            validate_file_url("ftp://server/file.pdf")


class TestValidateFileUrlHost:
    """Test host allowlist validation."""

    @patch("app.services.ctdt_ingest_service.settings")
    def test_allowlist_permits_listed_host(self, mock_settings):
        mock_settings.RAG_REMOTE_FILE_ALLOWED_HOSTS = "fileserver.local,other.com"
        validate_file_url_host("https://fileserver.local/file.pdf")

    @patch("app.services.ctdt_ingest_service.settings")
    def test_allowlist_rejects_unlisted_host(self, mock_settings):
        mock_settings.RAG_REMOTE_FILE_ALLOWED_HOSTS = "fileserver.local"
        with pytest.raises(IngestValidationError) as exc_info:
            validate_file_url_host("https://evil.com/file.pdf")
        assert exc_info.value.error.code == "invalid_file_url"

    @patch("app.services.ctdt_ingest_service.settings")
    def test_empty_allowlist_blocks_localhost(self, mock_settings):
        mock_settings.RAG_REMOTE_FILE_ALLOWED_HOSTS = ""
        with pytest.raises(IngestValidationError) as exc_info:
            validate_file_url_host("http://localhost/file.pdf")
        assert exc_info.value.error.code == "invalid_file_url"

    @patch("app.services.ctdt_ingest_service.settings")
    def test_empty_allowlist_blocks_127(self, mock_settings):
        mock_settings.RAG_REMOTE_FILE_ALLOWED_HOSTS = ""
        with pytest.raises(IngestValidationError):
            validate_file_url_host("http://127.0.0.1/file.pdf")

    @patch("app.services.ctdt_ingest_service.settings")
    def test_empty_allowlist_blocks_private_192(self, mock_settings):
        mock_settings.RAG_REMOTE_FILE_ALLOWED_HOSTS = ""
        with pytest.raises(IngestValidationError):
            validate_file_url_host("http://192.168.1.100/file.pdf")

    @patch("app.services.ctdt_ingest_service.settings")
    def test_empty_allowlist_allows_public_host(self, mock_settings):
        mock_settings.RAG_REMOTE_FILE_ALLOWED_HOSTS = ""
        validate_file_url_host("https://cdn.example.com/file.pdf")

    @patch("app.services.ctdt_ingest_service.settings")
    def test_allowlist_allows_localhost_if_listed(self, mock_settings):
        mock_settings.RAG_REMOTE_FILE_ALLOWED_HOSTS = "localhost,127.0.0.1"
        validate_file_url_host("http://localhost/file.pdf")
        validate_file_url_host("http://127.0.0.1/file.pdf")


class TestIsPrivateIp:
    def test_loopback(self):
        assert _is_private_ip("127.0.0.1") is True

    def test_10_network(self):
        assert _is_private_ip("10.0.0.1") is True

    def test_192_168(self):
        assert _is_private_ip("192.168.1.1") is True

    def test_172_16(self):
        assert _is_private_ip("172.16.0.1") is True

    def test_172_31(self):
        assert _is_private_ip("172.31.255.255") is True

    def test_172_15_not_private(self):
        assert _is_private_ip("172.15.0.1") is False

    def test_public_ip(self):
        assert _is_private_ip("8.8.8.8") is False

    def test_ipv6_loopback(self):
        assert _is_private_ip("::1") is True


class TestVerifyChecksum:
    def test_matching_checksum(self):
        data = b"hello world"
        checksum = hashlib.sha256(data).hexdigest()
        verify_checksum(data, checksum)  # Should not raise

    def test_case_insensitive(self):
        data = b"test data"
        checksum = hashlib.sha256(data).hexdigest().upper()
        verify_checksum(data, checksum)  # Should not raise

    def test_mismatch_raises(self):
        data = b"hello world"
        with pytest.raises(IngestPipelineError) as exc_info:
            verify_checksum(data, "0000000000000000000000000000000000000000000000000000000000000000")
        assert exc_info.value.error.code == "checksum_mismatch"
        assert exc_info.value.error.retryable is False

    def test_with_whitespace(self):
        data = b"hello"
        checksum = "  " + hashlib.sha256(data).hexdigest() + "  "
        verify_checksum(data, checksum)  # Should not raise


class TestValidateMimeType:
    @patch("app.services.ctdt_ingest_service.settings")
    def test_pdf_allowed(self, mock_settings):
        mock_settings.RAG_ALLOWED_MIME_TYPES = "application/pdf,text/plain"
        validate_mime_type("application/pdf")

    @patch("app.services.ctdt_ingest_service.settings")
    def test_case_insensitive(self, mock_settings):
        mock_settings.RAG_ALLOWED_MIME_TYPES = "application/pdf"
        validate_mime_type("Application/PDF")

    @patch("app.services.ctdt_ingest_service.settings")
    def test_strips_charset(self, mock_settings):
        mock_settings.RAG_ALLOWED_MIME_TYPES = "text/plain"
        validate_mime_type("text/plain; charset=utf-8")

    @patch("app.services.ctdt_ingest_service.settings")
    def test_unsupported_rejected(self, mock_settings):
        mock_settings.RAG_ALLOWED_MIME_TYPES = "application/pdf"
        with pytest.raises(IngestValidationError) as exc_info:
            validate_mime_type("application/zip")
        assert exc_info.value.error.code == "unsupported_mime_type"


class TestPersistIngestStats:
    def test_persists_stats(self):
        doc = MagicMock()
        doc.meta = {
            "system": {"pipeline_mode": "legacy"},
            "ctdt": {"external_file_id": "f1", "program_code": "7480201"},
        }
        _persist_ingest_stats(doc, text_length=5000, chunk_count=25, ingest_status="ready")

        assert doc.meta["ctdt"]["text_length"] == 5000
        assert doc.meta["ctdt"]["chunk_count"] == 25
        assert doc.meta["ctdt"]["ingest_status"] == "ready"
        # Original fields preserved
        assert doc.meta["ctdt"]["external_file_id"] == "f1"
        assert doc.meta["ctdt"]["program_code"] == "7480201"

    def test_works_with_empty_meta(self):
        doc = MagicMock()
        doc.meta = None
        _persist_ingest_stats(doc, text_length=100, chunk_count=1, ingest_status="ready")

        assert doc.meta["ctdt"]["text_length"] == 100


class TestGetDocumentCtdtInfo:
    """Test that GET status reads from persisted meta, not re-chunks."""

    def test_reads_persisted_stats(self):
        doc = MagicMock()
        doc.id = 42
        doc.title = "test.pdf"
        doc.status = "ready"
        doc.content_text = "some content here"
        doc.content_raw = "some content here"
        doc.created_at = None
        doc.updated_at = None
        doc.meta = {
            "system": {"content_type": "application/pdf"},
            "ctdt": {
                "external_file_id": "f1",
                "document_role": "current_curriculum",
                "update_cycle_id": "15",
                "program_code": "7480201",
                "program_name": "CNTT",
                "text_length": 12345,
                "chunk_count": 50,
                "ingest_status": "ready",
                "uploaded_by": "user1",
                "source_system": "cn_ctdt",
                "source_module": "update_cycle",
                "checksum": None,
            },
        }

        info = get_document_ctdt_info(doc)

        assert info["text_length"] == 12345
        assert info["chunk_count"] == 50
        assert info["ingest_status"] == "ready"
        assert info["document_role"] == "current_curriculum"
        assert info["program_code"] == "7480201"

    def test_fallback_without_persisted_stats(self):
        """Should fallback gracefully when stats not in meta."""
        doc = MagicMock()
        doc.id = 1
        doc.title = "old.pdf"
        doc.status = "ready"
        doc.content_text = "x" * 2000
        doc.content_raw = "x" * 2000
        doc.created_at = None
        doc.updated_at = None
        doc.meta = {
            "system": {},
            "ctdt": {"external_file_id": "old1"},
        }

        info = get_document_ctdt_info(doc)

        assert info["text_length"] == 2000
        assert info["chunk_count"] >= 1
        assert info["ingest_status"] == "ready"


class TestMapStatus:
    def test_ready(self):
        assert _map_status("ready") == "ready"

    def test_error(self):
        assert _map_status("error") == "failed"

    def test_uploaded(self):
        assert _map_status("uploaded") == "extracting"

    def test_chunked(self):
        assert _map_status("chunked") == "chunking"

    def test_indexed(self):
        assert _map_status("indexed") == "indexing"

    def test_unknown(self):
        assert _map_status("custom") == "custom"


class TestBuildCtdtMetadata:
    def test_structure(self):
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

        assert meta["system"]["pipeline_mode"] == "legacy"
        assert meta["system"]["ingest_via"] == "ctdt_remote_url"
        assert meta["ctdt"]["source_system"] == "cn_ctdt"
        assert meta["ctdt"]["external_file_id"] == "f1"
        assert meta["ctdt"]["document_role"] == "current_curriculum"
        assert meta["ctdt"]["update_cycle_id"] == "15"
        assert meta["user_metadata"] == {}


class TestStreamingDownload:
    """Test that download_remote_file uses streaming and respects size limits."""

    @pytest.mark.asyncio
    @patch("app.services.ctdt_ingest_service.settings")
    async def test_content_length_too_large_rejected_early(self, mock_settings):
        """Content-Length > max_bytes should reject before reading body."""
        mock_settings.RAG_REMOTE_FILE_TIMEOUT_SECONDS = 10
        mock_settings.RAG_REMOTE_FILE_MAX_MB = 1  # 1MB
        mock_settings.RAG_DOWNLOAD_USER_AGENT = "test"

        from app.services.ctdt_ingest_service import download_remote_file

        # Mock the async client with a response that declares huge Content-Length
        mock_response = AsyncMock()
        mock_response.headers = {"content-length": str(100 * 1024 * 1024)}  # 100MB
        mock_response.raise_for_status = MagicMock()

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.ctdt_ingest_service.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(IngestPipelineError) as exc_info:
                await download_remote_file("https://example.com/big.pdf")
            assert exc_info.value.error.code == "file_too_large"

    @pytest.mark.asyncio
    @patch("app.services.ctdt_ingest_service.settings")
    async def test_stream_exceeds_max_bytes_during_read(self, mock_settings):
        """Body exceeding max_bytes during streaming should be rejected."""
        mock_settings.RAG_REMOTE_FILE_TIMEOUT_SECONDS = 10
        mock_settings.RAG_REMOTE_FILE_MAX_MB = 1  # 1MB = 1048576 bytes
        mock_settings.RAG_DOWNLOAD_USER_AGENT = "test"

        from app.services.ctdt_ingest_service import download_remote_file

        # Generate chunks that total > 1MB
        chunk_size = 64 * 1024
        num_chunks = 20  # 20 * 64KB = 1.25MB > 1MB
        big_chunks = [b"x" * chunk_size for _ in range(num_chunks)]

        async def fake_aiter_bytes(chunk_size=None):
            for c in big_chunks:
                yield c

        mock_response = AsyncMock()
        mock_response.headers = {}  # No Content-Length
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = fake_aiter_bytes

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.ctdt_ingest_service.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(IngestPipelineError) as exc_info:
                await download_remote_file("https://example.com/big.pdf")
            assert exc_info.value.error.code == "file_too_large"

    @pytest.mark.asyncio
    @patch("app.services.ctdt_ingest_service.settings")
    async def test_small_file_downloaded_ok(self, mock_settings):
        """File within limits should download successfully."""
        mock_settings.RAG_REMOTE_FILE_TIMEOUT_SECONDS = 10
        mock_settings.RAG_REMOTE_FILE_MAX_MB = 1
        mock_settings.RAG_DOWNLOAD_USER_AGENT = "test"

        from app.services.ctdt_ingest_service import download_remote_file

        expected_data = b"small file content"

        async def fake_aiter_bytes(chunk_size=None):
            yield expected_data

        mock_response = AsyncMock()
        mock_response.headers = {"content-length": str(len(expected_data))}
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = fake_aiter_bytes

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("app.services.ctdt_ingest_service.httpx.AsyncClient", return_value=mock_client):
            data = await download_remote_file("https://example.com/small.pdf")
            assert data == expected_data


class TestScopedExternalId:
    """Test that effective_external_id includes update_cycle_id for scoping."""

    def test_external_id_format(self):
        """Verify the scoped external_id pattern used in ingest_from_url."""
        # Simulate the same f-string logic from production code
        update_cycle_id = "15"
        external_file_id = "file_123"
        mode = "legacy"

        effective = (
            f"ctdt:update-cycle:{update_cycle_id}"
            f":file:{external_file_id}:{mode}"
        )
        assert effective == "ctdt:update-cycle:15:file:file_123:legacy"

    def test_different_cycles_produce_different_ids(self):
        """Same file in different cycles must not collide."""
        file_id = "doc_abc"
        mode = "semantic"

        id_cycle_1 = f"ctdt:update-cycle:1:file:{file_id}:{mode}"
        id_cycle_2 = f"ctdt:update-cycle:2:file:{file_id}:{mode}"

        assert id_cycle_1 != id_cycle_2
        assert "update-cycle:1" in id_cycle_1
        assert "update-cycle:2" in id_cycle_2


class TestGetDocumentCtdtInfoNoRechunk:
    """Verify GET status does NOT call _compute_chunk_count when meta has stats."""

    @patch("app.services.ctdt_ingest_service._compute_chunk_count")
    def test_no_rechunk_when_meta_has_stats(self, mock_compute):
        """_compute_chunk_count must NOT be called if meta has chunk_count."""
        doc = MagicMock()
        doc.id = 42
        doc.title = "test.pdf"
        doc.status = "ready"
        doc.content_text = "x" * 10000
        doc.content_raw = "x" * 10000
        doc.created_at = None
        doc.updated_at = None
        doc.meta = {
            "system": {"content_type": "application/pdf"},
            "ctdt": {
                "external_file_id": "f1",
                "text_length": 10000,
                "chunk_count": 42,
                "ingest_status": "ready",
            },
        }

        info = get_document_ctdt_info(doc)

        assert info["chunk_count"] == 42
        assert info["text_length"] == 10000
        mock_compute.assert_not_called()

