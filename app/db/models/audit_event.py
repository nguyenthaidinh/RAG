"""
Audit event ORM model (Phase 6.0).

Append-only — NO update / delete methods are provided.
"""
from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[_uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), primary_key=True, default=_uuid.uuid4
    )

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )

    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    actor: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(Text, nullable=False)
    ref_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    ref_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
