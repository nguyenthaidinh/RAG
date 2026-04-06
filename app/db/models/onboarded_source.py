"""
Onboarded source ORM model (Phase 4 — Operationalize Source Platform).

Represents a single knowledge source onboarded into the AI Server.
Each row describes how to connect to, authenticate with, and fetch
content from one external API or system.

Schema is **generic** — no web-specific columns.  Different sources
are distinguished solely by their ``source_key`` + config columns.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class OnboardedSource(Base):
    __tablename__ = "onboarded_sources"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )

    tenant_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("tenants.id"),
        nullable=False,
    )

    # Runtime-stable identifier, maps to DocumentService.upsert(source=...)
    source_key: Mapped[str] = mapped_column(String(128), nullable=False)

    # Human-readable display name
    name: Mapped[str] = mapped_column(String(256), nullable=False)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Connector type — e.g. "internal-api", "database", "html"
    connector_type: Mapped[str] = mapped_column(String(64), nullable=False)

    # Base URL of the source API
    base_url: Mapped[str] = mapped_column(String(1024), nullable=False)

    # Authentication type: "bearer", "api_key", "none"
    auth_type: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'bearer'")
    )

    # Auth secrets — e.g. {"token": "..."}.  Masked on API read.
    auth_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    # Endpoint paths
    list_path: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        server_default=text("'/api/internal/knowledge/items'"),
    )
    detail_path_template: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )

    # Extra request config — timeouts, custom headers, etc.
    request_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    # Field mapping hints for canonical item mapping
    mapping_config: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    # Default params for list requests — e.g. {"kind": "policy", "status": "published"}
    default_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )

    last_synced_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # ── Phase 7: Scheduling fields ───────────────────────────────
    sync_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    sync_interval_minutes: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("60")
    )
    last_sync_attempt_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    next_sync_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # ── Phase 7: Health tracking fields ──────────────────────────
    last_success_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    last_failure_at = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    consecutive_failures: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    last_error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True
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
            "source_key",
            name="uq_onboarded_sources_tenant_source_key",
        ),
        Index("idx_onboarded_sources_tenant_id", "tenant_id"),
        Index("idx_onboarded_sources_connector_type", "connector_type"),
        Index("idx_onboarded_sources_is_active", "is_active"),
        Index(
            "idx_onboarded_sources_tenant_active",
            "tenant_id",
            "is_active",
        ),
    )
