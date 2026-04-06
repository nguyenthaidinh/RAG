"""
Source document link ORM model (Phase 8 — Delta-Aware Sync).

Maps source-side items to internal AI Server documents.
Each row tracks: identity, checksum for delta detection,
timestamps for seen/synced lifecycle, and status for missing sweep.

Schema is **generic** — no source-specific columns.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class SourceDocumentLink(Base):
    __tablename__ = "source_document_links"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.id"),
        nullable=False,
    )

    onboarded_source_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("onboarded_sources.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Denormalized for fast queries without joining onboarded_sources
    source_key: Mapped[str] = mapped_column(String(128), nullable=False)

    # Item identity in the source system
    external_id: Mapped[str] = mapped_column(String(512), nullable=False)

    # Link back to the source item (optional)
    external_uri: Mapped[str | None] = mapped_column(
        String(1024), nullable=True
    )

    # Internal document reference — BigInteger to match documents.id
    document_id: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True
    )

    # Tracks which document version this link corresponds to
    # Text to match documents.version_id type
    document_version_id: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    # Source-side content checksum for delta detection
    # String(64) to match documents.checksum type (SHA-256 hex)
    content_checksum: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )

    # Source-side last-modified timestamp
    remote_updated_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Updated on every run where this item appears in source
    last_seen_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Updated only when content was actually ingested/re-ingested
    last_synced_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # active, missing, error
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'active'"),
    )

    # Lightweight tracking metadata (e.g. error info, sync notes)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    created_at = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "onboarded_source_id",
            "external_id",
            name="uq_source_doc_links_tenant_source_ext",
        ),
        Index(
            "idx_source_doc_links_tenant_source",
            "tenant_id",
            "onboarded_source_id",
        ),
        Index(
            "idx_source_doc_links_tenant_source_status",
            "tenant_id",
            "onboarded_source_id",
            "status",
        ),
        Index(
            "idx_source_doc_links_document_id",
            "document_id",
        ),
    )
