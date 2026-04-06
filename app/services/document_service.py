# app/services/document_service.py
import hashlib
import logging
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.repos.document_repo import DocumentRepo
from app.services.document_event_emitter import (
    DOCUMENT_CREATED,
    DOCUMENT_STATUS_CHANGED,
    DOCUMENT_UPDATED,
    emit_document_event,
)
from app.services.document_lifecycle import (
    CHUNKED,
    ERROR,
    INDEXED,
    READY,
    UPLOADED,
    validate_transition,
)
from app.services.exceptions import QuotaExceededError
from app.services.vector_index import NullIndex, PgVectorIndex, VectorIndex

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Orchestrates the full document ingest pipeline::

        raw -> clean -> chunk -> meter -> embed -> index -> ready

    Caller owns transaction lifecycle; this service only flushes.
    """

    def __init__(
        self,
        repo: DocumentRepo | None = None,
        *,
        embedding_provider=None,
        vector_index: VectorIndex | None = None,
    ):
        self.repo = repo or DocumentRepo()
        self._embedding_provider = embedding_provider
        self._vector_index = vector_index

    @property
    def embedding_provider(self):
        if self._embedding_provider is None:
            from app.services.embedding_provider import get_embedding_provider

            self._embedding_provider = get_embedding_provider()
        return self._embedding_provider

    @property
    def vector_index(self) -> VectorIndex:
        if self._vector_index is None:
            from app.services.vector_index import get_vector_index

            self._vector_index = get_vector_index()
        return self._vector_index

    @staticmethod
    def _checksum(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_metadata(metadata: dict | None) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}
        return metadata

    @staticmethod
    def _normalize_title(title: str | None) -> str | None:
        if title is None:
            return None
        title = title.strip()
        return title or None

    @staticmethod
    def _get_pipeline_mode(metadata: dict | None) -> str | None:
        if not isinstance(metadata, dict):
            return None
        system_meta = metadata.get("system") or {}
        value = system_meta.get("pipeline_mode")
        return str(value) if value else None

    @staticmethod
    def _should_reprocess_content(
        *,
        existing_doc: Document,
        new_checksum: str,
    ) -> bool:
        if existing_doc.checksum != new_checksum:
            return True
        if existing_doc.status != READY:
            return True
        return False

    @staticmethod
    def _process_content(
        raw_content: str,
        *,
        tenant_id: str,
        document_id: int,
        version_id: str,
    ) -> tuple[str, list]:
        from app.nlp import get_cleaner, get_chunker

        t0 = time.monotonic()

        cleaner = get_cleaner()
        chunker = get_chunker()

        cleaned = cleaner.clean(raw_content)
        chunks = chunker.chunk(
            cleaned,
            tenant_id=tenant_id,
            document_id=document_id,
            version_id=version_id,
        )

        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)
        total_tokens = sum(c.token_count for c in chunks)
        logger.info(
            "nlp.processed tenant_id=%s doc_id=%s chunks=%d tokens_total=%d elapsed_ms=%.2f",
            tenant_id,
            document_id,
            len(chunks),
            total_tokens,
            elapsed_ms,
        )
        return cleaned, chunks

    async def check_quota(
        self,
        *,
        db: AsyncSession,
        tenant_id: str,
        content_size_bytes: int,
    ) -> None:
        """
        Placeholder quota check.
        Replace with real tenant/document/token quota enforcement later.
        """
        allowed = True
        if not allowed:
            raise QuotaExceededError("Quota exceeded")

    async def _embed_and_index(
        self,
        db: AsyncSession,
        *,
        doc: Document,
        chunks: list,
        old_version: str | None,
    ) -> None:
        t0 = time.monotonic()

        try:
            if isinstance(self.vector_index, NullIndex):
                old_status = doc.status
                validate_transition(doc.status, READY)
                doc.status = READY
                await db.flush()

                if old_status != READY:
                    await emit_document_event(
                        db,
                        tenant_id=doc.tenant_id,
                        document_id=doc.id,
                        event_type=DOCUMENT_STATUS_CHANGED,
                        from_status=old_status,
                        to_status=READY,
                    )

                logger.info(
                    "ingest.ready tenant_id=%s doc_id=%s version_id=%s vector_index=null elapsed_ms=%.2f",
                    doc.tenant_id,
                    doc.id,
                    doc.checksum,
                    round((time.monotonic() - t0) * 1000, 2),
                )
                return

            texts = [c.text for c in chunks]
            embeddings = await self.embedding_provider.embed(texts)

            if old_version:
                if isinstance(self.vector_index, PgVectorIndex):
                    await self.vector_index.delete_in_tx(
                        db,
                        tenant_id=doc.tenant_id,
                        document_id=doc.id,
                        version_id=old_version,
                    )
                else:
                    await self.vector_index.delete(
                        tenant_id=doc.tenant_id,
                        document_id=doc.id,
                        version_id=old_version,
                    )

            if isinstance(self.vector_index, PgVectorIndex):
                await self.vector_index.upsert_in_tx(
                    db,
                    tenant_id=doc.tenant_id,
                    document_id=doc.id,
                    version_id=doc.checksum,
                    chunks=chunks,
                    embeddings=embeddings,
                )
            else:
                await self.vector_index.upsert(
                    tenant_id=doc.tenant_id,
                    document_id=doc.id,
                    version_id=doc.checksum,
                    chunks=chunks,
                    embeddings=embeddings,
                )

            elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

            old_status = doc.status
            validate_transition(doc.status, INDEXED)
            doc.status = INDEXED
            await db.flush()

            if old_status != INDEXED:
                await emit_document_event(
                    db,
                    tenant_id=doc.tenant_id,
                    document_id=doc.id,
                    event_type=DOCUMENT_STATUS_CHANGED,
                    from_status=old_status,
                    to_status=INDEXED,
                )

            old_status = doc.status
            validate_transition(doc.status, READY)
            doc.status = READY
            await db.flush()

            if old_status != READY:
                await emit_document_event(
                    db,
                    tenant_id=doc.tenant_id,
                    document_id=doc.id,
                    event_type=DOCUMENT_STATUS_CHANGED,
                    from_status=old_status,
                    to_status=READY,
                )

            logger.info(
                "ingest.indexed tenant_id=%s doc_id=%s version_id=%s embedding_count=%d chunk_count=%d elapsed_ms=%.2f",
                doc.tenant_id,
                doc.id,
                doc.checksum,
                len(embeddings),
                len(chunks),
                elapsed_ms,
            )

        except Exception:
            logger.exception(
                "ingest.embed_index_failed tenant_id=%s doc_id=%s version_id=%s",
                doc.tenant_id,
                doc.id,
                doc.checksum,
            )
            try:
                old_status = doc.status
                validate_transition(doc.status, ERROR)
                doc.status = ERROR
                await db.flush()

                if old_status != ERROR:
                    await emit_document_event(
                        db,
                        tenant_id=doc.tenant_id,
                        document_id=doc.id,
                        event_type=DOCUMENT_STATUS_CHANGED,
                        from_status=old_status,
                        to_status=ERROR,
                    )
            except Exception:
                logger.exception("ingest.error_status_failed doc_id=%s", doc.id)

    async def upsert(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source: str,
        external_id: str,
        content: str,
        title: str | None,
        metadata: dict | None,
        representation_type: str = "original",
        parent_document_id: int | None = None,
    ) -> tuple[Document, str, bool]:
        title = self._normalize_title(title)
        metadata = self._normalize_metadata(metadata)
        pipeline_mode = self._get_pipeline_mode(metadata)

        checksum = self._checksum(content)
        content_size = len(content.encode("utf-8"))

        logger.info(
            "document.upsert.start tenant_id=%s source=%s external_id=%s pipeline_mode=%s representation_type=%s",
            tenant_id,
            source,
            external_id,
            pipeline_mode,
            representation_type,
        )

        await self.check_quota(
            db=db,
            tenant_id=tenant_id,
            content_size_bytes=content_size,
        )

        doc = await self.repo.get_by_key(
            db,
            tenant_id=tenant_id,
            source=source,
            external_id=external_id,
        )

        # ── CREATE ───────────────────────────────────────────────
        if not doc:
            doc = Document(
                tenant_id=tenant_id,
                source=source,
                external_id=external_id,
                title=title,
                content_raw=content,
                content_text="",
                meta=metadata,
                checksum=checksum,
                version_id=checksum,
                status=UPLOADED,
                representation_type=representation_type,
                parent_document_id=parent_document_id,
            )
            self.repo.add(db, doc)
            await db.flush()

            await emit_document_event(
                db,
                tenant_id=tenant_id,
                document_id=doc.id,
                event_type=DOCUMENT_CREATED,
                to_status=UPLOADED,
                message=f"Document created via source={source}",
                metadata_json={
                    "source": source,
                    "representation_type": representation_type,
                    "pipeline_mode": pipeline_mode,
                },
            )

            cleaned, chunks = self._process_content(
                content,
                tenant_id=tenant_id,
                document_id=doc.id,
                version_id=checksum,
            )
            doc.content_text = cleaned

            old_status = doc.status
            validate_transition(doc.status, CHUNKED)
            doc.status = CHUNKED
            await db.flush()

            await emit_document_event(
                db,
                tenant_id=tenant_id,
                document_id=doc.id,
                event_type=DOCUMENT_STATUS_CHANGED,
                from_status=old_status,
                to_status=CHUNKED,
            )

            await self._embed_and_index(db, doc=doc, chunks=chunks, old_version=None)

            logger.info(
                "document.upsert.created tenant_id=%s doc_id=%s external_id=%s pipeline_mode=%s",
                tenant_id,
                doc.id,
                external_id,
                pipeline_mode,
            )
            return doc, "created", True

        # ── EXISTING ─────────────────────────────────────────────
        content_changed = doc.checksum != checksum
        needs_reprocess = self._should_reprocess_content(
            existing_doc=doc,
            new_checksum=checksum,
        )

        title_changed = (title != doc.title)
        meta_changed = (metadata != (doc.meta or {}))
        representation_changed = (representation_type != doc.representation_type)
        parent_changed = (parent_document_id != doc.parent_document_id)

        # ── FAST UPDATE / NOOP: content unchanged + READY ────────
        if not needs_reprocess:
            if title_changed:
                doc.title = title
            if meta_changed:
                doc.meta = metadata
            if representation_changed:
                doc.representation_type = representation_type
            if parent_changed:
                doc.parent_document_id = parent_document_id

            # Backfill version_id for legacy rows created before this fix
            version_backfilled = False
            if not doc.version_id:
                doc.version_id = doc.checksum
                version_backfilled = True

            if title_changed or meta_changed or representation_changed or parent_changed or version_backfilled:
                await db.flush()

                await emit_document_event(
                    db,
                    tenant_id=tenant_id,
                    document_id=doc.id,
                    event_type=DOCUMENT_UPDATED,
                    message="Metadata/title/representation updated (content unchanged)",
                    metadata_json={
                        "content_changed": False,
                        "title_changed": title_changed,
                        "meta_changed": meta_changed,
                        "representation_changed": representation_changed,
                        "parent_changed": parent_changed,
                        "pipeline_mode": pipeline_mode,
                        "reprocessed": False,
                        "version_backfilled": version_backfilled,
                    },
                )

                logger.info(
                    "document.upsert.updated_no_reprocess tenant_id=%s doc_id=%s external_id=%s pipeline_mode=%s version_backfilled=%s",
                    tenant_id,
                    doc.id,
                    external_id,
                    pipeline_mode,
                    version_backfilled,
                )
                return doc, "updated", True

            logger.info(
                "document.upsert.noop tenant_id=%s doc_id=%s external_id=%s pipeline_mode=%s",
                tenant_id,
                doc.id,
                external_id,
                pipeline_mode,
            )
            return doc, "noop", False

        # ── UPDATE / RETRY WITH REPROCESS ────────────────────────
        old_version: str | None = None
        if content_changed:
            old_version = doc.checksum
            doc.checksum = checksum
            doc.content_raw = content

        # Always sync version_id with current checksum on reprocess
        doc.version_id = checksum

        if title_changed:
            doc.title = title
        if meta_changed:
            doc.meta = metadata
        if representation_changed:
            doc.representation_type = representation_type
        if parent_changed:
            doc.parent_document_id = parent_document_id

        cleaned, chunks = self._process_content(
            content,
            tenant_id=tenant_id,
            document_id=doc.id,
            version_id=checksum,
        )
        doc.content_text = cleaned

        old_status = doc.status
        if doc.status != CHUNKED:
            validate_transition(doc.status, CHUNKED)
            doc.status = CHUNKED

        await db.flush()

        await emit_document_event(
            db,
            tenant_id=tenant_id,
            document_id=doc.id,
            event_type=DOCUMENT_UPDATED,
            message="Document content updated",
            metadata_json={
                "content_changed": content_changed,
                "title_changed": title_changed,
                "meta_changed": meta_changed,
                "representation_changed": representation_changed,
                "parent_changed": parent_changed,
                "pipeline_mode": pipeline_mode,
                "reprocessed": True,
            },
        )

        if old_status != CHUNKED:
            await emit_document_event(
                db,
                tenant_id=tenant_id,
                document_id=doc.id,
                event_type=DOCUMENT_STATUS_CHANGED,
                from_status=old_status,
                to_status=CHUNKED,
            )

        await self._embed_and_index(
            db,
            doc=doc,
            chunks=chunks,
            old_version=old_version,
        )

        logger.info(
            "document.upsert.updated_reprocessed tenant_id=%s doc_id=%s external_id=%s pipeline_mode=%s content_changed=%s",
            tenant_id,
            doc.id,
            external_id,
            pipeline_mode,
            content_changed,
        )
        return doc, "updated", True


