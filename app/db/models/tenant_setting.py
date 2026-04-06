"""
Tenant settings ORM model (Phase 5.0).

Per-tenant plan assignment and quota overrides.
"""
from __future__ import annotations

from sqlalchemy import Boolean, String, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.models.user import Base


class TenantSetting(Base):
    __tablename__ = "tenant_settings"

    tenant_id: Mapped[str] = mapped_column(String(64), primary_key=True)

    plan_code: Mapped[str] = mapped_column(Text, nullable=False, server_default="free")
    quota_overrides_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    enforce_user_rate_limit: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false",
    )

    created_at = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
