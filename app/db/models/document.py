from __future__ import annotations

from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Index, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    external_id: Mapped[str] = mapped_column(String(512), nullable=False)

    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_raw: Mapped[str] = mapped_column(Text, nullable=False)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # ORM attr is "meta" to avoid collision with DeclarativeBase.metadata
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )

    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    # immutable content version for vector integrity validation (Phase 8.x)
    version_id: Mapped[str | None] = mapped_column(Text, nullable=True, index=True)

    # lifecycle: pending -> processing -> ready | error
    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        server_default=text("'pending'"),
        index=True,
    )

    # 'original' = ingested document, 'synthesized' = AI-generated derivative
    representation_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'original'"),
    )

    # Self-FK: if synthesized, points to the source original document
    parent_document_id: Mapped[int | None] = mapped_column(
        BigInteger,
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
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
            "source",
            "external_id",
            name="uq_documents_tenant_source_external",
        ),
        CheckConstraint(
            "status IN ('uploaded', 'pending', 'chunked', 'processing', "
            "'indexed', 'ready', 'error')",
            name="ck_documents_status",
        ),
        CheckConstraint(
            "representation_type IN ('original', 'synthesized')",
            name="ck_documents_representation_type",
        ),
        Index("idx_documents_tenant_status", "tenant_id", "status"),
        Index("idx_documents_tenant_source_external", "tenant_id", "source", "external_id"),
        Index("idx_documents_parent_document_id", "parent_document_id"),
        Index("idx_documents_tenant_parent_repr", "tenant_id", "parent_document_id", "representation_type"),
    )