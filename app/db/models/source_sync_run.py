"""
Source sync run ORM model (Phase 4 — Operationalize Source Platform).

Tracks each sync execution for auditing, debugging, and operational
visibility.  One row per sync invocation.
"""
from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class SourceSyncRun(Base):
    __tablename__ = "source_sync_runs"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)

    source_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("onboarded_sources.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Denormalized for fast queries without joins
    source_key: Mapped[str] = mapped_column(String(128), nullable=False)

    # running → success | failed
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'running'"),
    )

    started_at = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    finished_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    items_fetched: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    items_upserted: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    items_failed: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    # Phase 8: Granular delta metrics
    items_created: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    items_updated: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    items_unchanged: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    items_missing: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    items_reactivated: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # "manual", "api", or future "scheduler"
    triggered_by: Mapped[str | None] = mapped_column(
        String(32), nullable=True
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
        Index("idx_sync_runs_tenant_id", "tenant_id"),
        Index("idx_sync_runs_source_id", "source_id"),
        Index("idx_sync_runs_status", "status"),
        Index("idx_sync_runs_started_at", "started_at"),
        Index(
            "idx_sync_runs_tenant_source",
            "tenant_id",
            "source_id",
        ),
    )
