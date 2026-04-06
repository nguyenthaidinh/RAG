"""
Query usage ORM model (Phase 4.1).

Tracks every retrieval query for billing, auditing, and analytics.
Privacy-safe: stores only query_hash (SHA-256), NEVER raw query text.
"""
from __future__ import annotations

import uuid as _uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class QueryUsage(Base):
    __tablename__ = "query_usages"

    id: Mapped[_uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)

    idempotency_key: Mapped[str] = mapped_column(String(512), nullable=False)
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    query_len: Mapped[int] = mapped_column(Integer, nullable=False)

    mode: Mapped[str] = mapped_column(String(20), nullable=False)  # vector|bm25|hybrid

    k_final: Mapped[int] = mapped_column(Integer, nullable=False)
    k_vector: Mapped[int | None] = mapped_column(Integer, nullable=True)
    k_bm25: Mapped[int | None] = mapped_column(Integer, nullable=True)

    results_count: Mapped[int] = mapped_column(Integer, nullable=False)

    tokens_query: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_context: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_total: Mapped[int] = mapped_column(Integer, nullable=False)

    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "idempotency_key",
            name="uq_query_usages_tenant_idempotency",
        ),
    )
