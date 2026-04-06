import asyncio
import os
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.db.models.user import User
from app.db.models.tenant import Tenant
from app.core.security import hash_password
from app.core.rbac import ROLE_SYSTEM_ADMIN


DEFAULT_TENANT_ID = "default"
DEFAULT_TENANT_NAME = "Default Tenant"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env: {name}")
    return value


async def ensure_default_tenant(session) -> Tenant:
    result = await session.execute(
        select(Tenant).where(Tenant.id == DEFAULT_TENANT_ID)
    )
    tenant = result.scalar_one_or_none()

    if not tenant:
        tenant = Tenant(
            id=DEFAULT_TENANT_ID,
            name=DEFAULT_TENANT_NAME,
            is_active=True,
            max_users=100,
        )
        session.add(tenant)
        await session.flush()
        print(f"✅ Created default tenant: {DEFAULT_TENANT_ID}")

    return tenant


async def main():
    email = _require_env("ADMIN_EMAIL")
    password = _require_env("ADMIN_PASSWORD")

    async with AsyncSessionLocal() as session:
        tenant = await ensure_default_tenant(session)

        result = await session.execute(
            select(User).where(User.email == email)
        )
        user = result.scalar_one_or_none()

        if user:
            # ⚠️ SAFE GUARD: do not reset password silently
            if user.role != ROLE_SYSTEM_ADMIN:
                user.role = ROLE_SYSTEM_ADMIN
                user.is_active = True
                user.tenant_id = tenant.id
                print("🔁 Updated user role to SYSTEM_ADMIN")
            else:
                print("ℹ️  Admin already exists – no password change")
        else:
            user = User(
                email=email,
                password_hash=hash_password(password),
                role=ROLE_SYSTEM_ADMIN,
                is_active=True,
                tenant_id=tenant.id,
            )
            session.add(user)
            print("✅ Created system admin user")

        await session.commit()

    print("👉 Admin bootstrap completed.")


if __name__ == "__main__":
    asyncio.run(main())
