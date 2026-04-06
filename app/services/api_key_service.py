import secrets
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timezone
from app.db.models.api_key import APIKey
from app.core.security import hash_password, verify_password
from app.core.audit import audit_log


class APIKeyService:
    @staticmethod
    async def get_by_id(db: AsyncSession, key_id: int) -> APIKey | None:
        res = await db.execute(select(APIKey).where(APIKey.id == key_id))
        return res.scalar_one_or_none()

    @staticmethod
    async def list_by_user_id(db: AsyncSession, user_id: int) -> list[APIKey]:
        res = await db.execute(
            select(APIKey)
            .where(APIKey.user_id == user_id)
            .order_by(APIKey.created_at.desc())
        )
        return list(res.scalars().all())

    @staticmethod
    async def verify_api_key(
        db: AsyncSession,
        full_key: str,
    ) -> APIKey | None:
        """
        Verify API key and update last_used_at if valid.
        Returns APIKey object if valid, None otherwise.

        Format: ak_live_<prefix>.<secret>
        Lookup by prefix, verify secret against secret_hash (argon2, constant-time).
        """
        if "." not in full_key:
            return None

        prefix, secret = full_key.split(".", 1)
        if not prefix or len(prefix) < 4:
            return None
        if not secret:
            return None

        res = await db.execute(
            select(APIKey).where(
                APIKey.prefix == prefix,
                APIKey.is_active == True,
                APIKey.revoked_at.is_(None),
            )
        )
        api_key = res.scalar_one_or_none()
        if not api_key:
            return None

        # Constant-time hash verification (argon2)
        if not verify_password(secret, api_key.secret_hash):
            return None

        # Update last_used_at
        api_key.last_used_at = datetime.now(timezone.utc)
        await db.flush()

        return api_key

    @staticmethod
    async def revoke_api_key(
        db: AsyncSession,
        key_id: int,
        *,
        actor_user_id: int | None = None,
    ) -> APIKey | None:
        """Revoke an API key by setting revoked_at and is_active=False."""
        api_key = await APIKeyService.get_by_id(db, key_id)
        if not api_key:
            return None

        api_key.revoked_at = datetime.now(timezone.utc)
        api_key.is_active = False
        await db.flush()

        audit_log(
            action="api_key.revoke",
            actor_user_id=actor_user_id,
            tenant_id=api_key.tenant_id,
            target_id=api_key.id,
            detail=f"Revoked API key '{api_key.name}' (prefix={api_key.prefix})",
        )

        return api_key

    @staticmethod
    async def rotate_api_key(
        db: AsyncSession,
        key_id: int,
        *,
        actor_user_id: int | None = None,
    ) -> tuple[APIKey, str] | None:
        """
        Revoke old key and create new one for the same user+tenant.
        Fully transactional: wrapped in a SAVEPOINT so that if creation
        fails, the revocation is also rolled back.
        Returns (new_api_key_obj, new_full_key_string).
        """
        old_key = await APIKeyService.get_by_id(db, key_id)
        if not old_key:
            return None

        # Atomic: revoke + create inside a single savepoint
        async with db.begin_nested():
            old_key.revoked_at = datetime.now(timezone.utc)
            old_key.is_active = False

            # Inline key creation (prefix.secret format)
            raw_secret = secrets.token_urlsafe(32)
            prefix = f"ak_live_{secrets.token_hex(8)}"
            new_key = APIKey(
                user_id=old_key.user_id,
                tenant_id=old_key.tenant_id,
                name=f"{old_key.name} (rotated)",
                prefix=prefix,
                secret_hash=hash_password(raw_secret),
                is_active=True,
            )
            db.add(new_key)
            await db.flush()
            full_key = f"{prefix}.{raw_secret}"

        audit_log(
            action="api_key.rotate",
            actor_user_id=actor_user_id,
            tenant_id=old_key.tenant_id,
            target_id=old_key.id,
            detail=f"Rotated API key '{old_key.name}' → new id={new_key.id}",
        )

        return new_key, full_key
