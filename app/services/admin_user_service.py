from __future__ import annotations

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.audit import audit_log
from app.core.rbac import (
    ROLE_TENANT_ADMIN,
    TENANT_ADMIN_ASSIGNABLE_ROLES,
    validate_role,
    is_system_admin,
)
from app.core.security import hash_password
from app.db.models.api_key import APIKey
from app.db.models.quota import UserQuota
from app.db.models.tenant import Tenant
from app.db.models.usage import UsageLedger
from app.db.models.user import User


# ===== Default quota (can be replaced by plan-based logic later)
DEFAULT_QUOTA = {
    "max_tokens": 1_000_000,
    "max_requests": 100_000,
    "max_storage_mb": 1024,
}

# Guardrails
MAX_PAGE_SIZE = 200
MAX_SEARCH_LEN = 256


class AdminUserService:
    # =========================
    # SEARCH USERS
    # =========================
    @staticmethod
    async def search_users(
        db: AsyncSession,
        *,
        actor: User | None = None,
        q: str | None = None,
        status: str = "all",
        page: int = 1,
        page_size: int = 50,
    ) -> tuple[list[User], int]:
        """
        Search users.
        system_admin sees all users.
        tenant_admin sees only users in their own tenant.
        """
        # Defensive pagination
        page = max(1, int(page or 1))
        page_size = int(page_size or 50)
        page_size = min(max(1, page_size), MAX_PAGE_SIZE)

        status = (status or "all").strip().lower()

        query = select(User)

        # Tenant scoping for non-system_admin
        if actor and not is_system_admin(actor.role):
            query = query.where(User.tenant_id == actor.tenant_id)

        if q:
            q_norm = q.strip()
            if q_norm:
                q_norm = q_norm[:MAX_SEARCH_LEN]
                query = query.where(User.email.ilike(f"%{q_norm}%"))

        if status == "active":
            query = query.where(User.is_active.is_(True))
        elif status == "inactive":
            query = query.where(User.is_active.is_(False))
        # else: "all" => no filter

        count_query = select(func.count()).select_from(query.subquery())
        total = (await db.execute(count_query)).scalar_one() or 0

        query = (
            query.order_by(User.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )

        users = (await db.execute(query)).scalars().all()
        return list(users), total

    # =========================
    # CREATE USER (CORE)
    # =========================
    @staticmethod
    async def create_user(
        db: AsyncSession,
        *,
        email: str,
        password: str,
        tenant_id: str,
        role: str = "user",
        is_active: bool = True,
        actor: User | None = None,
    ) -> User:
        """
        Create new user within a tenant.
        Validates:
        - email uniqueness
        - tenant exists & is active
        - tenant max_users not exceeded
        - RBAC: actor can create user with the requested role in the target tenant

        Does NOT commit – caller is responsible.
        """
        # Validate role
        validate_role(role)

        # RBAC enforcement
        if actor:
            _enforce_user_creation_rbac(actor, tenant_id, role)

        email_norm = (email or "").strip().lower()
        if not email_norm:
            raise ValueError("Email is required")

        # Check email uniqueness
        exists = await db.execute(select(User.id).where(User.email == email_norm))
        if exists.scalar_one_or_none():
            raise ValueError("Email already exists")

        # Validate tenant
        tenant = (await db.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()
        if not tenant:
            raise ValueError(f"Tenant '{tenant_id}' does not exist")
        if not tenant.is_active:
            raise ValueError(f"Tenant '{tenant_id}' is inactive")

        # Enforce max_users
        current_count = (
            await db.execute(select(func.count(User.id)).where(User.tenant_id == tenant_id))
        ).scalar_one() or 0
        if tenant.max_users > 0 and current_count >= tenant.max_users:
            raise ValueError(
                f"Tenant '{tenant_id}' has reached its user limit ({tenant.max_users})"
            )

        user = User(
            email=email_norm,
            password_hash=hash_password(password),
            role=role,
            is_active=is_active,
            tenant_id=tenant_id,
        )
        db.add(user)
        await db.flush()  # get user.id

        quota = UserQuota(
            user_id=user.id,
            max_tokens=DEFAULT_QUOTA["max_tokens"],
            max_requests=DEFAULT_QUOTA["max_requests"],
            max_storage_mb=DEFAULT_QUOTA["max_storage_mb"],
        )
        db.add(quota)

        # ✅ Privacy-safe audit (no raw email in logs)
        audit_log(
            action="user.create",
            actor_user_id=actor.id if actor else None,
            tenant_id=tenant_id,
            target_id=user.id,
            detail=f"Created user target_id={user.id} role={role} is_active={is_active}",
        )

        return user

    # =========================
    # USER DETAIL
    # =========================
    @staticmethod
    async def get_user_detail(
        db: AsyncSession,
        user_id: int,
        *,
        actor: User | None = None,
    ) -> dict | None:
        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if not user:
            return None

        # Tenant scoping for non-system_admin
        if actor and not is_system_admin(actor.role):
            if user.tenant_id != actor.tenant_id:
                return None

        quota = (await db.execute(select(UserQuota).where(UserQuota.user_id == user_id))).scalar_one_or_none()

        api_keys_count = (
            await db.execute(select(func.count(APIKey.id)).where(APIKey.user_id == user_id))
        ).scalar_one() or 0

        recent_usage = (
            await db.execute(
                select(UsageLedger)
                .where(UsageLedger.user_id == user_id)
                .order_by(UsageLedger.created_at.desc())
                .limit(20)
            )
        ).scalars().all()

        return {
            "user": user,
            "quota": quota,
            "api_keys_count": api_keys_count,
            "recent_usage": list(recent_usage),
        }

    # =========================
    # UPDATE ROLE
    # =========================
    @staticmethod
    async def update_user_role(
        db: AsyncSession,
        user_id: int,
        role: str,
        *,
        actor: User | None = None,
    ) -> User | None:
        validate_role(role)

        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if not user:
            return None

        # RBAC enforcement
        if actor:
            _enforce_role_change_rbac(actor, user, role)

        old_role = user.role
        user.role = role

        audit_log(
            action="user.role_change",
            actor_user_id=actor.id if actor else None,
            tenant_id=user.tenant_id,
            target_id=user.id,
            detail=f"Changed role old={old_role} new={role}",
        )

        return user

    # =========================
    # TOGGLE STATUS
    # =========================
    @staticmethod
    async def toggle_user_status(
        db: AsyncSession,
        user_id: int,
        is_active: bool,
        *,
        actor: User | None = None,
    ) -> User | None:
        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if not user:
            return None

        # Tenant scoping for non-system_admin
        if actor and not is_system_admin(actor.role):
            if user.tenant_id != actor.tenant_id:
                return None

        user.is_active = bool(is_active)

        audit_log(
            action="user.status_change",
            actor_user_id=actor.id if actor else None,
            tenant_id=user.tenant_id,
            target_id=user.id,
            detail=f"Set is_active={bool(is_active)}",
        )

        return user


# ── RBAC helper functions (private) ───────────────────────────────

def _enforce_user_creation_rbac(actor: User, tenant_id: str, role: str) -> None:
    """
    Enforce RBAC rules for user creation.
    Raises ValueError on violation.
    """
    if is_system_admin(actor.role):
        # system_admin can create any user in any tenant with any role
        return

    if actor.role == ROLE_TENANT_ADMIN:
        # tenant_admin can only create users in their own tenant
        if actor.tenant_id != tenant_id:
            raise ValueError("Tenant admin can only create users in their own tenant")
        # tenant_admin cannot create system_admin
        if role not in TENANT_ADMIN_ASSIGNABLE_ROLES:
            raise ValueError(
                f"Tenant admin cannot assign role '{role}'. "
                f"Allowed: {', '.join(sorted(TENANT_ADMIN_ASSIGNABLE_ROLES))}"
            )
        return

    # Regular users cannot create other users
    raise ValueError("Insufficient permissions to create users")


def _enforce_role_change_rbac(actor: User, target: User, new_role: str) -> None:
    """
    Enforce RBAC rules for role changes.
    Raises ValueError on violation.
    """
    if is_system_admin(actor.role):
        # system_admin can change any role
        return

    if actor.role == ROLE_TENANT_ADMIN:
        # Must be in same tenant
        if target.tenant_id != actor.tenant_id:
            raise ValueError("Tenant admin can only modify users in their own tenant")
        # Cannot promote to system_admin
        if new_role not in TENANT_ADMIN_ASSIGNABLE_ROLES:
            raise ValueError(
                f"Tenant admin cannot assign role '{new_role}'. "
                f"Allowed: {', '.join(sorted(TENANT_ADMIN_ASSIGNABLE_ROLES))}"
            )
        # Cannot modify another tenant admin (only system_admin can)
        if target.role == ROLE_TENANT_ADMIN and target.id != actor.id:
            raise ValueError("Tenant admin cannot modify another tenant admin's role")
        return

    raise ValueError("Insufficient permissions to change user roles")
