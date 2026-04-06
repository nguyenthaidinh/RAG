"""
Rate-limit bucket ORM model (Phase 5.0).

DB-backed token bucket for multi-process-safe rate limiting.
"""
from __future__ import annotations

import uuid as _uuid
from datetime import datetime

from sqlalchemy import BigInteger, Integer, String, Text, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class RateLimitBucket(Base):
    __tablename__ = "rate_limit_buckets"

    id: Mapped[_uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)

    scope: Mapped[str] = mapped_column(Text, nullable=False)           # "tenant" | "user"
    window_sec: Mapped[int] = mapped_column(Integer, nullable=False)   # e.g. 60
    bucket_key: Mapped[str] = mapped_column(Text, nullable=False)      # e.g. "qpm"

    tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    reset_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    updated_at = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "user_id", "scope", "bucket_key", "window_sec",
            name="uq_rate_limit_bucket",
        ),
        Index(
            "ix_rate_limit_scope_reset",
            "tenant_id", "user_id", "scope", "reset_at",
        ),
    )
