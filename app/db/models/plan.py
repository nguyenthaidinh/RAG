"""
Plan ORM model (Phase 5.0).

Represents a pricing/quota plan that tenants can be assigned to.
Business pricing can be decided later — the schema is intentionally flexible.
"""
from __future__ import annotations

import uuid as _uuid

from sqlalchemy import Boolean, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[_uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )

    code: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")

    limits_json: Mapped[dict] = mapped_column(JSONB, nullable=False)

    created_at = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
