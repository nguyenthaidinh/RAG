from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean, BigInteger, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.sql import func
from app.db.models.user import Base


class UserQuota(Base):
    __tablename__ = "user_quotas"

    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), primary_key=True)
    
    plan: Mapped[str] = mapped_column(String(50), nullable=False, server_default="free")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    
    max_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1000000")
    used_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    
    max_requests: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1000")
    used_requests: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    
    max_storage_mb: Mapped[int] = mapped_column(Integer, nullable=False, server_default="100")
    used_storage_mb: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    
    created_at = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
