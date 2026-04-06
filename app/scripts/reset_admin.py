import asyncio
from sqlalchemy import select
from passlib.context import CryptContext

from app.db.session import AsyncSessionLocal
from app.db.models.user import User
from app.db.models.tenant import Tenant

pwd = CryptContext(schemes=["argon2"], deprecated="auto")


async def reset_admin():
    async with AsyncSessionLocal() as db:
        email = "admin@ai.server"
        password = "123456"
        role = "system_admin"

        # Lấy tenant đầu tiên nếu đã có
        tenant_result = await db.execute(select(Tenant).limit(1))
        tenant = tenant_result.scalar_one_or_none()

        # Nếu chưa có tenant thì tạo tenant mặc định
        if tenant is None:
            tenant = Tenant(
                id="default",
                name="Default Tenant",
                is_active=True,
                max_users=100,
            )
            db.add(tenant)
            await db.flush()  # để tenant có thể dùng ngay trong cùng transaction
            print("Created default tenant: default")

        # Tìm user admin theo email
        user_result = await db.execute(select(User).where(User.email == email))
        user = user_result.scalar_one_or_none()

        if user:
            user.password_hash = pwd.hash(password)
            user.is_active = True
            user.role = role
            user.tenant_id = tenant.id
            print(f"Password reset: {email} / {password}")
        else:
            user = User(
                email=email,
                password_hash=pwd.hash(password),
                role=role,
                tenant_id=tenant.id,
                is_active=True,
            )
            db.add(user)
            print(f"Admin created: {email} / {password}")

        await db.commit()


if __name__ == "__main__":
    asyncio.run(reset_admin())