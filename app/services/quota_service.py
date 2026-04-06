from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.quota import UserQuota
from app.db.models.tenant_quota import TenantQuota
from app.db.models.usage import UsageLedger
from app.services.exceptions import QuotaExceededError
from app.core.audit import audit_log
from datetime import datetime, timedelta, timezone
import logging

logger = logging.getLogger(__name__)


class QuotaService:
    # ── User-level quota (kept for backward compat / per-user admin) ──

    @staticmethod
    async def get_by_user_id(db: AsyncSession, user_id: int) -> UserQuota | None:
        res = await db.execute(select(UserQuota).where(UserQuota.user_id == user_id))
        return res.scalar_one_or_none()

    @staticmethod
    async def update_quota(
        db: AsyncSession,
        user_id: int,
        plan: str | None = None,
        max_tokens: int | None = None,
        max_requests: int | None = None,
        max_storage_mb: int | None = None,
        is_active: bool | None = None,
    ) -> UserQuota | None:
        quota = await QuotaService.get_by_user_id(db, user_id)
        if not quota:
            return None

        if plan is not None:
            quota.plan = plan
        if max_tokens is not None:
            quota.max_tokens = max_tokens
        if max_requests is not None:
            quota.max_requests = max_requests
        if max_storage_mb is not None:
            quota.max_storage_mb = max_storage_mb
        if is_active is not None:
            quota.is_active = is_active

        await db.flush()
        return quota

    @staticmethod
    async def set_unlimited(db: AsyncSession, user_id: int) -> UserQuota | None:
        quota = await QuotaService.get_by_user_id(db, user_id)
        if not quota:
            return None

        quota.max_tokens = -1
        quota.max_requests = -1
        quota.max_storage_mb = -1
        await db.flush()
        return quota

    @staticmethod
    async def get_usage_aggregates(
        db: AsyncSession,
        user_id: int,
        days: int = 7,
    ) -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=days)

        res = await db.execute(
            select(
                func.count(UsageLedger.id).label("request_count"),
                func.sum(UsageLedger.tokens_total).label("tokens_total"),
                func.sum(UsageLedger.file_size_bytes).label("storage_bytes"),
            )
            .where(
                UsageLedger.user_id == user_id,
                UsageLedger.created_at >= since,
            )
        )
        row = res.first()

        return {
            "request_count": row.request_count or 0,
            "tokens_total": int(row.tokens_total or 0),
            "storage_bytes": int(row.storage_bytes or 0),
            "storage_mb": int((row.storage_bytes or 0) / 1024 / 1024),
            "days": days,
        }

    # ── Tenant-level quota (NEW – atomic enforcement) ──────────────

    @staticmethod
    async def get_tenant_quota(db: AsyncSession, tenant_id: str) -> TenantQuota | None:
        res = await db.execute(
            select(TenantQuota).where(TenantQuota.tenant_id == tenant_id)
        )
        return res.scalar_one_or_none()

    @staticmethod
    async def create_tenant_quota(
        db: AsyncSession,
        tenant_id: str,
        *,
        plan: str = "free",
        max_requests: int = 10_000,
        max_tokens: int = 10_000_000,
        max_storage_mb: int = 1024,
    ) -> TenantQuota:
        """Create a tenant quota record. Caller must commit."""
        tq = TenantQuota(
            tenant_id=tenant_id,
            plan=plan,
            is_active=True,
            max_requests=max_requests,
            max_tokens=max_tokens,
            max_storage_mb=max_storage_mb,
        )
        db.add(tq)
        await db.flush()
        return tq

    @staticmethod
    async def update_tenant_quota(
        db: AsyncSession,
        tenant_id: str,
        *,
        plan: str | None = None,
        max_requests: int | None = None,
        max_tokens: int | None = None,
        max_storage_mb: int | None = None,
        is_active: bool | None = None,
    ) -> TenantQuota | None:
        tq = await QuotaService.get_tenant_quota(db, tenant_id)
        if not tq:
            return None

        if plan is not None:
            tq.plan = plan
        if max_requests is not None:
            tq.max_requests = max_requests
        if max_tokens is not None:
            tq.max_tokens = max_tokens
        if max_storage_mb is not None:
            tq.max_storage_mb = max_storage_mb
        if is_active is not None:
            tq.is_active = is_active

        await db.flush()
        return tq

    # ── ATOMIC enforcement (single-SQL, no race condition) ─────────

    @staticmethod
    async def atomic_increment_request(
        db: AsyncSession,
        tenant_id: str,
    ) -> bool:
        """
        Atomically increment used_requests for a tenant.
        Returns True if the increment succeeded (quota not exceeded).
        Returns False if quota is exceeded or inactive.

        Uses a single UPDATE … WHERE to prevent race conditions.
        No SELECT+UPDATE pattern.
        """
        result = await db.execute(
            text("""
                UPDATE tenant_quotas
                SET used_requests = used_requests + 1,
                    updated_at = NOW()
                WHERE tenant_id = :tenant_id
                  AND is_active = true
                  AND (max_requests = -1 OR used_requests < max_requests)
                RETURNING used_requests
            """),
            {"tenant_id": tenant_id},
        )
        row = result.fetchone()
        if row is None:
            # Either tenant_quota doesn't exist, is inactive, or quota exceeded
            return False
        return True

    @staticmethod
    async def atomic_increment_tokens(
        db: AsyncSession,
        tenant_id: str,
        tokens: int,
    ) -> bool:
        """Atomically increment used_tokens for a tenant."""
        if tokens <= 0:
            return True

        result = await db.execute(
            text("""
                UPDATE tenant_quotas
                SET used_tokens = used_tokens + :tokens,
                    updated_at = NOW()
                WHERE tenant_id = :tenant_id
                  AND is_active = true
                  AND (max_tokens = -1 OR used_tokens + :tokens <= max_tokens)
                RETURNING used_tokens
            """),
            {"tenant_id": tenant_id, "tokens": tokens},
        )
        row = result.fetchone()
        return row is not None

    @staticmethod
    async def atomic_increment_storage(
        db: AsyncSession,
        tenant_id: str,
        storage_mb: int,
    ) -> bool:
        """Atomically increment used_storage_mb for a tenant."""
        if storage_mb <= 0:
            return True

        result = await db.execute(
            text("""
                UPDATE tenant_quotas
                SET used_storage_mb = used_storage_mb + :storage_mb,
                    updated_at = NOW()
                WHERE tenant_id = :tenant_id
                  AND is_active = true
                  AND (max_storage_mb = -1 OR used_storage_mb + :storage_mb <= max_storage_mb)
                RETURNING used_storage_mb
            """),
            {"tenant_id": tenant_id, "storage_mb": storage_mb},
        )
        row = result.fetchone()
        return row is not None

    @staticmethod
    async def check_and_enforce_quota(
        db: AsyncSession,
        tenant_id: str,
    ) -> tuple[bool, str | None]:
        """
        Atomically check and enforce tenant-level request quota.
        Returns (allowed: bool, error_message: str | None).

        This is the PRIMARY quota enforcement gate.
        Uses single atomic SQL UPDATE – no race conditions.
        """
        # Try atomic increment
        allowed = await QuotaService.atomic_increment_request(db, tenant_id)
        if not allowed:
            # Determine reason for better error message
            tq = await QuotaService.get_tenant_quota(db, tenant_id)
            if not tq:
                msg = "Tenant quota not configured"
            elif not tq.is_active:
                msg = "Tenant quota is inactive"
            else:
                msg = "Request quota exceeded"

            audit_log(
                action="quota.exceeded",
                tenant_id=tenant_id,
                detail=msg,
            )
            return False, msg

        return True, None

    # ── Legacy compat shim: check_and_enforce_quota by user_id ─────

    @staticmethod
    async def check_and_enforce_quota_by_user(
        db: AsyncSession,
        user_id: int,
        tenant_id: str,
    ) -> tuple[bool, str | None]:
        """
        Backward-compatible wrapper that routes to tenant-level enforcement.
        Falls back to user-level check if no tenant_quota exists.
        """
        tq = await QuotaService.get_tenant_quota(db, tenant_id)
        if tq:
            return await QuotaService.check_and_enforce_quota(db, tenant_id)

        # Fallback: user-level quota (legacy)
        quota = await QuotaService.get_by_user_id(db, user_id)
        if not quota:
            return False, "Quota not found"
        if not quota.is_active:
            return False, "Quota is inactive"
        if quota.max_requests != -1 and quota.used_requests >= quota.max_requests:
            return False, "Request quota exceeded"
        return True, None

    @staticmethod
    async def increment_usage(
        db: AsyncSession,
        user_id: int,
        tokens_total: int = 0,
        storage_bytes: int = 0,
        requests: int = 1,
    ) -> None:
        """
        Increment user-level quota counters (legacy).
        For atomic tenant-level enforcement, use atomic_increment_* methods.
        """
        quota = await QuotaService.get_by_user_id(db, user_id)
        if not quota:
            return

        if quota.max_tokens != -1:
            quota.used_tokens += tokens_total
        if quota.max_requests != -1:
            quota.used_requests += requests
        if quota.max_storage_mb != -1:
            quota.used_storage_mb += int(storage_bytes / 1024 / 1024)

        await db.flush()
