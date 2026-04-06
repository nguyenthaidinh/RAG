"""
Document admin service (Phase 2A).

Read/admin orchestration layer for document management.
Separated from DocumentService (write/ingest) by design.

Responsibilities:
  - List / detail / preview / download / history
  - Content truncation for previews
  - Download payload construction
  - Privacy-safe: no raw content logging
  
Rules:
  - All operations tenant-scoped
  - Preview truncation enforced
  - No mutation of documents
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.repos.document_admin_repo import DocumentAdminRepo, DocumentListFilters

logger = logging.getLogger(__name__)

# Maximum characters in a preview (safety ceiling)
PREVIEW_MAX_CHARS = 20_000


@dataclass(frozen=True)
class OpenPayload:
    """Result of an open/preview operation."""

    document_id: int
    view: str
    title: str | None
    mime_type: str
    content: str
    truncated: bool
    full_length: int
    shown_length: int


@dataclass(frozen=True)
class DownloadPayload:
    """Result of a download operation."""

    document_id: int
    view: str
    format: str
    filename: str
    mime_type: str
    content: str


@dataclass(frozen=True)
class ContentStats:
    """Content size statistics for a document."""

    raw_length: int
    text_length: int
    has_raw: bool
    has_text: bool
    content_type: str | None


class DocumentAdminService:
    """
    Admin service for document read operations.

    All methods require tenant_id for scoping.
    No mutations — Phase 2B will add action methods.
    """

    def __init__(self, repo: DocumentAdminRepo | None = None) -> None:
        self._repo = repo or DocumentAdminRepo()

    async def get_list(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        q: str | None = None,
        status: str | None = None,
        representation_type: str | None = None,
        source: str | None = None,
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[dict], int]:
        """
        List documents with filters.

        Returns (items_dicts, total_count).
        """
        filters = DocumentListFilters(
            q=q,
            status=status,
            representation_type=representation_type,
            source=source,
        )

        docs, total = await self._repo.list_documents(
            db,
            tenant_id=tenant_id,
            filters=filters,
            page=page,
            page_size=page_size,
        )

        items = [self._doc_to_list_item(d) for d in docs]
        return items, total

    async def get_detail(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
    ) -> dict | None:
        """
        Get full document detail for admin view.

        Returns None if not found (for 404, not for cross-tenant — tenant
        scoping is done at the repo layer).
        """
        doc = await self._repo.get_document_by_id(
            db, document_id=document_id, tenant_id=tenant_id,
        )
        if not doc:
            return None

        related = await self._repo.get_related_representations(
            db, document_id=document_id, tenant_id=tenant_id,
        )

        events = await self._repo.list_document_events(
            db, document_id=document_id, tenant_id=tenant_id, limit=10,
        )

        return {
            "document": self._doc_to_detail(doc),
            "metadata": self._extract_metadata(doc),
            "content_stats": self._build_content_stats(doc),
            "related": {
                "parent": (
                    self._doc_to_list_item(related["parent"])
                    if related["parent"]
                    else None
                ),
                "children": [
                    self._doc_to_list_item(c) for c in related["children"]
                ],
            },
            "history_summary": [
                self._event_to_dict(e) for e in events
            ],
        }

    async def get_open_payload(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
        view: str = "cleaned",
        max_chars: int = PREVIEW_MAX_CHARS,
    ) -> OpenPayload | None:
        """
        Get preview content for a document.

        view: "raw" | "cleaned" | "synthesized"
        """
        doc = await self._repo.get_document_by_id(
            db, document_id=document_id, tenant_id=tenant_id,
        )
        if not doc:
            return None

        content, mime_type = self._resolve_view_content(doc, view)
        full_length = len(content) if content else 0
        truncated = full_length > max_chars

        if truncated and content:
            content = content[:max_chars]

        return OpenPayload(
            document_id=document_id,
            view=view,
            title=doc.title,
            mime_type=mime_type,
            content=content or "",
            truncated=truncated,
            full_length=full_length,
            shown_length=len(content) if content else 0,
        )

    async def build_download_payload(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
        view: str = "cleaned",
        format: str = "txt",
    ) -> DownloadPayload | None:
        """
        Build download content for a document.

        view: "raw" | "cleaned" | "synthesized"
        format: "txt" | "md" | "json"
        """
        doc = await self._repo.get_document_by_id(
            db, document_id=document_id, tenant_id=tenant_id,
        )
        if not doc:
            return None

        content, _ = self._resolve_view_content(doc, view)
        if not content:
            content = ""

        # Build output by format
        if format == "json":
            export = {
                "document_id": doc.id,
                "tenant_id": doc.tenant_id,
                "title": doc.title,
                "source": doc.source,
                "external_id": doc.external_id,
                "status": doc.status,
                "representation_type": doc.representation_type,
                "parent_document_id": doc.parent_document_id,
                "checksum": doc.checksum,
                "view": view,
                "content": content,
                "metadata": self._safe_metadata(doc.meta),
                "created_at": (
                    doc.created_at.isoformat() if doc.created_at else None
                ),
                "updated_at": (
                    doc.updated_at.isoformat() if doc.updated_at else None
                ),
            }
            output = json.dumps(export, ensure_ascii=False, indent=2)
            mime_type = "application/json"
            ext = "json"
        elif format == "md":
            title_line = f"# {doc.title}\n\n" if doc.title else ""
            output = f"{title_line}{content}"
            mime_type = "text/markdown"
            ext = "md"
        else:
            output = content
            mime_type = "text/plain"
            ext = "txt"

        # Build filename
        safe_title = (doc.title or f"document_{doc.id}")[:60]
        safe_title = "".join(
            c if c.isalnum() or c in ("-", "_", " ") else "_"
            for c in safe_title
        ).strip()
        filename = f"{safe_title}_{view}.{ext}"

        return DownloadPayload(
            document_id=document_id,
            view=view,
            format=format,
            filename=filename,
            mime_type=mime_type,
            content=output,
        )

    async def get_history(
        self,
        db: AsyncSession,
        *,
        document_id: int,
        tenant_id: str,
        limit: int = 50,
    ) -> list[dict] | None:
        """
        Get document event history, newest first.

        Returns None if document not found.
        """
        doc = await self._repo.get_document_by_id(
            db, document_id=document_id, tenant_id=tenant_id,
        )
        if not doc:
            return None

        events = await self._repo.list_document_events(
            db, document_id=document_id, tenant_id=tenant_id, limit=limit,
        )

        return [self._event_to_dict(e) for e in events]

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _doc_to_list_item(doc: Document) -> dict:
        """Convert a Document to a list-friendly dict."""
        meta = doc.meta or {}
        return {
            "id": doc.id,
            "tenant_id": doc.tenant_id,
            "title": doc.title,
            "source": doc.source,
            "external_id": doc.external_id,
            "status": doc.status,
            "representation_type": doc.representation_type,
            "parent_document_id": doc.parent_document_id,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "content_length": len(doc.content_raw) if doc.content_raw else 0,
            "content_type": meta.get("content_type") or meta.get("file_type"),
        }

    @staticmethod
    def _doc_to_detail(doc: Document) -> dict:
        """Convert a Document to a detail dict."""
        return {
            "id": doc.id,
            "tenant_id": doc.tenant_id,
            "title": doc.title,
            "source": doc.source,
            "external_id": doc.external_id,
            "status": doc.status,
            "representation_type": doc.representation_type,
            "parent_document_id": doc.parent_document_id,
            "checksum": doc.checksum,
            "version_id": doc.version_id,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
        }

    @staticmethod
    def _extract_metadata(doc: Document) -> dict:
        """Extract display-safe metadata from a document."""
        meta = doc.meta or {}
        return {
            "file_name": meta.get("file_name"),
            "content_type": meta.get("content_type") or meta.get("file_type"),
            "size_bytes": meta.get("size_bytes"),
            "ingest_via": meta.get("ingest_via"),
            "checksum": doc.checksum,
            "version_id": doc.version_id,
        }

    @staticmethod
    def _build_content_stats(doc: Document) -> dict:
        """Build content size statistics."""
        meta = doc.meta or {}
        return {
            "raw_length": len(doc.content_raw) if doc.content_raw else 0,
            "text_length": len(doc.content_text) if doc.content_text else 0,
            "has_raw": bool(doc.content_raw),
            "has_text": bool(doc.content_text),
            "content_type": meta.get("content_type") or meta.get("file_type"),
        }

    @staticmethod
    def _resolve_view_content(
        doc: Document,
        view: str,
    ) -> tuple[str | None, str]:
        """
        Resolve which content field to use for a given view.

        Returns (content, mime_type).
        """
        if view == "raw":
            return doc.content_raw, "text/plain"
        elif view == "synthesized":
            # For synthesized docs, content_text IS the synthesized content.
            # For original docs with synthesized children, we don't have it here.
            if doc.representation_type == "synthesized":
                return doc.content_text or doc.content_raw, "text/markdown"
            return None, "text/plain"
        else:
            # "cleaned" — default
            return doc.content_text or doc.content_raw, "text/plain"

    @staticmethod
    def _safe_metadata(meta: dict | None) -> dict:
        """Return metadata safe for JSON export (no content leakage)."""
        if not meta:
            return {}
        # Exclude keys that might contain raw content
        excluded = {"content_raw", "content_text", "raw_content"}
        return {k: v for k, v in meta.items() if k not in excluded}

    @staticmethod
    def _event_to_dict(event) -> dict:
        """Convert a DocumentEvent to a serializable dict."""
        return {
            "id": str(event.id),
            "event_type": event.event_type,
            "from_status": event.from_status,
            "to_status": event.to_status,
            "actor_user_id": event.actor_user_id,
            "request_id": event.request_id,
            "message": event.message,
            "metadata_json": event.metadata_json or {},
            "created_at": event.created_at,
        }
