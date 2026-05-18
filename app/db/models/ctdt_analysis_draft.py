from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, Index, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class CTDTAnalysisDraft(Base):
    """Persisted CTDT analysis draft, isolated from official program tables."""

    __tablename__ = "ctdt_analysis_drafts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    update_cycle_id: Mapped[str] = mapped_column(String(64), nullable=False)
    program_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    program_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    program_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    analysis_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    draft_type: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        server_default=text("'update_cycle_analysis'"),
    )
    result_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    source_summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_by: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    updated_by: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default=text("'draft'"),
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'archived')",
            name="ck_ctdt_analysis_drafts_status",
        ),
        Index(
            "idx_ctdt_analysis_drafts_tenant_cycle",
            "tenant_id",
            "update_cycle_id",
        ),
        Index(
            "idx_ctdt_analysis_drafts_tenant_cycle_program",
            "tenant_id",
            "update_cycle_id",
            "program_code",
        ),
        Index(
            "idx_ctdt_analysis_drafts_tenant_type",
            "tenant_id",
            "draft_type",
        ),
        Index(
            "idx_ctdt_analysis_drafts_latest",
            "tenant_id",
            "update_cycle_id",
            "analysis_mode",
            "updated_at",
        ),
        Index(
            "idx_ctdt_analysis_drafts_latest_lookup",
            "tenant_id",
            "update_cycle_id",
            "program_code",
            "analysis_mode",
            "draft_type",
            "updated_at",
            "id",
        ),
    )
