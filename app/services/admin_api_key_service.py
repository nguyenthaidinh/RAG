import secrets
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from app.db.models.api_key import APIKey
from app.db.models.user import User
from app.core.security import hash_password
from app.core.audit import audit_log


class AdminAPIKeyService:
    @staticmethod
    def generate_api_key() -> tuple[str, str, str]:
        """
        Generate API key in format: ak_live_<prefix>.<secret>
        Returns (full_key, prefix, secret)

        The prefix stored in DB MUST equal full_key.split(".", 1)[0]
        so that APIKeyService.verify_api_key() can look it up correctly.

        ONLY the secret is hashed (argon2) and stored.
        The full key is NEVER persisted.
        """
        prefix_part = secrets.token_urlsafe(8)[:8]
        secret = secrets.token_urlsafe(32)
        prefix = f"ak_live_{prefix_part}"
        full_key = f"{prefix}.{secret}"
        return full_key, prefix, secret

    @staticmethod
    async def list_api_keys(
        db: AsyncSession,
        user_id: int | None = None,
        tenant_id: str | None = None,
    ) -> list[dict]:
        """
        List API keys with user info.
        Optionally filter by user_id and/or tenant_id.
        """
        query = select(APIKey, User.email, User.role).join(User, APIKey.user_id == User.id)

        if user_id:
            query = query.where(APIKey.user_id == user_id)
        if tenant_id:
            query = query.where(APIKey.tenant_id == tenant_id)

        query = query.order_by(APIKey.created_at.desc())

        res = await db.execute(query)
        results = []
        for api_key, email, role in res.all():
            results.append({
                "id": api_key.id,
                "user_id": api_key.user_id,
                "tenant_id": api_key.tenant_id,
                "user_email": email,
                "user_role": role,
                "name": api_key.name,
                "prefix": api_key.prefix,
                "is_active": api_key.is_active,
                "revoked_at": api_key.revoked_at.isoformat() if api_key.revoked_at else None,
                "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                "created_at": api_key.created_at.isoformat(),
            })
        return results

    @staticmethod
    async def create_api_key(
        db: AsyncSession,
        user_id: int,
        tenant_id: str,
        name: str,
        *,
        actor_user_id: int | None = None,
    ) -> tuple[APIKey, str]:
        """
        Create API key.
        Hashes ONLY the secret part (argon2).
        Returns (api_key_obj, full_key_string).
        The full key is returned ONCE and never stored.
        """
        full_key, prefix, secret = AdminAPIKeyService.generate_api_key()
        secret_hash = hash_password(secret)

        api_key = APIKey(
            user_id=user_id,
            tenant_id=tenant_id,
            name=name,
            secret_hash=secret_hash,
            prefix=prefix,
            is_active=True,
        )
        db.add(api_key)
        await db.flush()

        audit_log(
            action="api_key.create",
            actor_user_id=actor_user_id,
            tenant_id=tenant_id,
            target_id=api_key.id,
            detail=f"Created API key '{name}' for user_id={user_id} (prefix={prefix})",
        )

        return api_key, full_key

    @staticmethod
    async def rename_api_key(
        db: AsyncSession,
        key_id: int,
        name: str,
    ) -> APIKey | None:
        """
        Rename API key.
        """
        api_key = await db.execute(select(APIKey).where(APIKey.id == key_id))
        api_key = api_key.scalar_one_or_none()
        if not api_key:
            return None

        api_key.name = name
        await db.flush()
        return api_key
