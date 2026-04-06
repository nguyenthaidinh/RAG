from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.sql import func
from app.db.models.user import Base


class APIKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(64), ForeignKey("tenants.id"), nullable=False)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    secret_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    prefix: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)

    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    revoked_at = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    last_used_at = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    created_at = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
