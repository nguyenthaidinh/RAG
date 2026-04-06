from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String, Boolean, Integer
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.sql import func
from app.db.models.user import Base


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    max_users: Mapped[int] = mapped_column(Integer, nullable=False, server_default="10")

    created_at = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
