"""
Document event ORM model (Phase 2A).

Append-only timeline of document lifecycle events.
Separate from audit_events — this is a domain-specific,
structured event log for document admin/history views.

Design rules:
  - Append-only: no UPDATE / DELETE
  - No raw content in metadata_json
  - Tenant-scoped for defense-in-depth
  - event_type is a constrained enum-like field
"""
from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import BigInteger, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class DocumentEvent(Base):
    __tablename__ = "document_events"

    id: Mapped[_uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    document_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)

    from_status: Mapped[str | None] = mapped_column(String(50), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(50), nullable=True)

    actor_user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_document_events_tenant_id", "tenant_id"),
        Index("ix_document_events_document_id", "document_id"),
        Index("ix_document_events_tenant_document", "tenant_id", "document_id"),
        Index("ix_document_events_event_type", "event_type"),
        Index("ix_document_events_created_at", "created_at"),
    )
