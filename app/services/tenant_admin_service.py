"""
Tenant administration service.

Provides minimal CRUD for the system_admin dashboard:
  - list_tenants   – search / filter / paginate
  - get_tenant_detail – tenant info + user count
  - create_tenant  – validate, persist, audit
"""
from __future__ import annotations

import re

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import audit_log
from app.db.models.tenant import Tenant
from app.db.models.user import User

# Guardrails
MAX_PAGE_SIZE = 200
MAX_SEARCH_LEN = 256

# Tenant ID format: lowercase alphanumeric, underscore, hyphen
_TENANT_ID_PATTERN = re.compile(r"^[a-z0-9_-]+$")


class TenantAdminService:
    """Stateless service layer for tenant management (system_admin only)."""

    # =========================
    # LIST TENANTS
    # =========================
    @staticmethod
    async def list_tenants(
        db: AsyncSession,
        *,
        q: str | None = None,
        status: str = "all",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[Tenant], int]:
        """
        Search tenants with optional text filter and status filter.

        Returns (items, total).
        """
        page = max(1, int(page or 1))
        page_size = int(page_size or 50)
        page_size = min(max(1, page_size), MAX_PAGE_SIZE)

        status = (status or "all").strip().lower()

        query = select(Tenant)

        # Text filter: match against id or name
        if q:
            q_norm = q.strip()[:MAX_SEARCH_LEN]
            if q_norm:
                query = query.where(
                    Tenant.id.ilike(f"%{q_norm}%")
                    | Tenant.name.ilike(f"%{q_norm}%")
                )

        if status == "active":
            query = query.where(Tenant.is_active.is_(True))
        elif status == "inactive":
            query = query.where(Tenant.is_active.is_(False))

        # Total count
        count_query = select(func.count()).select_from(query.subquery())
        total = (await db.execute(count_query)).scalar_one() or 0

        # Ordered results with pagination
        query = (
            query.order_by(Tenant.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        items = (await db.execute(query)).scalars().all()
        return list(items), total

    # =========================
    # TENANT DETAIL
    # =========================
    @staticmethod
    async def get_tenant_detail(
        db: AsyncSession,
        tenant_id: str,
    ) -> dict | None:
        """
        Return tenant info plus user count.

        Returns None if tenant not found.
        """
        tenant = (
            await db.execute(select(Tenant).where(Tenant.id == tenant_id))
        ).scalar_one_or_none()

        if not tenant:
            return None

        users_count = (
            await db.execute(
                select(func.count(User.id)).where(User.tenant_id == tenant_id)
            )
        ).scalar_one() or 0

        return {
            "tenant": tenant,
            "users_count": users_count,
        }

    # =========================
    # CREATE TENANT
    # =========================
    @staticmethod
    async def create_tenant(
        db: AsyncSession,
        *,
        id: str,
        name: str,
        max_users: int = 10,
        is_active: bool = True,
        actor: User | None = None,
    ) -> Tenant:
        """
        Create a new tenant.

        Validates inputs, checks for duplicate id.
        Flushes but does NOT commit – caller is responsible.

        Raises ValueError on validation failure.
        """
        # ── Normalize & validate id ──
        tid = (id or "").strip().lower()

        if not tid:
            raise ValueError("Tenant ID is required")
        if len(tid) < 3:
            raise ValueError("Tenant ID must be at least 3 characters")
        if len(tid) > 64:
            raise ValueError("Tenant ID must be at most 64 characters")
        if not _TENANT_ID_PATTERN.match(tid):
            raise ValueError(
                "Tenant ID may only contain lowercase letters, digits, "
                "underscores, and hyphens"
            )

        # ── Validate name ──
        name_clean = (name or "").strip()
        if not name_clean:
            raise ValueError("Tenant name is required")
        if len(name_clean) > 255:
            raise ValueError("Tenant name must be at most 255 characters")

        # ── Validate max_users ──
        max_users = int(max_users)
        if max_users < 0:
            raise ValueError("Max users must be 0 or greater")

        # ── Check duplicate ──
        existing = (
            await db.execute(select(Tenant.id).where(Tenant.id == tid))
        ).scalar_one_or_none()
        if existing is not None:
            raise ValueError(f"Tenant ID '{tid}' already exists")

        # ── Create ──
        tenant = Tenant(
            id=tid,
            name=name_clean,
            max_users=max_users,
            is_active=is_active,
        )
        db.add(tenant)
        await db.flush()

        # ── Audit (privacy-safe: no sensitive data) ──
        audit_log(
            action="tenant.create",
            actor_user_id=actor.id if actor else None,
            tenant_id=tid,
            target_id=tid,
            detail=f"Created tenant target_id={tid} max_users={max_users} is_active={is_active}",
        )

        return tenant
