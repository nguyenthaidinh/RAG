from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean, BigInteger, Integer, ForeignKey, Numeric, UniqueConstraint
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.sql import func
from app.db.models.user import Base


class UsageLedger(Base):
    __tablename__ = "usage_ledger"

    # Idempotency: UNIQUE(tenant_id, request_id) prevents double-log on retries.
    __table_args__ = (
        UniqueConstraint("tenant_id", "request_id", name="uq_usage_tenant_request"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    api_key_id = mapped_column(BigInteger, ForeignKey("api_keys.id"), nullable=True)
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    endpoint: Mapped[str] = mapped_column(String(512), nullable=False)
    method: Mapped[str] = mapped_column(String(10), nullable=False)

    tokens_input: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    tokens_output: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    tokens_total: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")

    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    request_cost: Mapped[float] = mapped_column(Numeric(10, 6), nullable=False, server_default="0")

    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_at = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
