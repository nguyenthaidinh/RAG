from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.user import User
from app.db.models.tenant import Tenant
from app.db.models.quota import UserQuota
from app.core.security import hash_password
from app.core.rbac import validate_role


class UserService:
    @staticmethod
    async def get_by_email(db: AsyncSession, email: str) -> User | None:
        res = await db.execute(select(User).where(User.email == email))
        return res.scalar_one_or_none()

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: int) -> User | None:
        res = await db.execute(select(User).where(User.id == user_id))
        return res.scalar_one_or_none()

    @staticmethod
    async def list_users(
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100,
    ) -> list[User]:
        res = await db.execute(
            select(User)
            .order_by(User.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(res.scalars().all())

    @staticmethod
    async def count_users(db: AsyncSession) -> int:
        res = await db.execute(select(func.count(User.id)))
        return res.scalar_one() or 0

    @staticmethod
    async def create_user(
        db: AsyncSession,
        email: str,
        password: str,
        tenant_id: str,
        role: str = "user",
    ) -> User:
        """
        Create a user within an existing tenant.
        Validates tenant exists, is active, and has capacity (max_users).
        Does NOT commit – caller is responsible for commit/rollback.
        """
        # Validate role
        validate_role(role)

        # Validate tenant
        tenant = (
            await db.execute(select(Tenant).where(Tenant.id == tenant_id))
        ).scalar_one_or_none()
        if not tenant:
            raise ValueError(f"Tenant '{tenant_id}' does not exist")
        if not tenant.is_active:
            raise ValueError(f"Tenant '{tenant_id}' is inactive")

        # Enforce max_users
        current_count = (
            await db.execute(
                select(func.count(User.id)).where(User.tenant_id == tenant_id)
            )
        ).scalar_one() or 0
        if tenant.max_users > 0 and current_count >= tenant.max_users:
            raise ValueError(
                f"Tenant '{tenant_id}' has reached its user limit ({tenant.max_users})"
            )

        user = User(
            email=email,
            password_hash=hash_password(password),
            role=role,
            tenant_id=tenant_id,
            is_active=True,
        )
        db.add(user)
        await db.flush()

        # Create initial quota
        quota = UserQuota(
            user_id=user.id,
            plan="free",
            is_active=True,
            max_tokens=1000000,
            max_requests=1000,
            max_storage_mb=100,
        )
        db.add(quota)
        await db.flush()

        return user

    @staticmethod
    async def set_user_status(
        db: AsyncSession,
        user_id: int,
        is_active: bool,
    ) -> User | None:
        user = await UserService.get_by_id(db, user_id)
        if not user:
            return None
        user.is_active = is_active
        await db.flush()
        return user
