from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.sql import func
from app.db.models.user import Base


class TenantQuota(Base):
    __tablename__ = "tenant_quotas"

    tenant_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("tenants.id"), primary_key=True
    )

    plan: Mapped[str] = mapped_column(String(50), nullable=False, server_default="free")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")

    max_requests: Mapped[int] = mapped_column(Integer, nullable=False, server_default="10000")
    used_requests: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="10000000")
    used_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    max_storage_mb: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1024")
    used_storage_mb: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    created_at = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
