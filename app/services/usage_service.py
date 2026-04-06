from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from app.db.models.usage import UsageLedger
import logging

logger = logging.getLogger(__name__)


class UsageService:
    @staticmethod
    async def log_usage(
        db: AsyncSession,
        user_id: int,
        tenant_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        success: bool,
        request_id: str,
        api_key_id: int | None = None,
        tokens_input: int = 0,
        tokens_output: int = 0,
        tokens_total: int = 0,
        file_size_bytes: int = 0,
        request_cost: float = 0.0,
    ) -> bool:
        """
        Insert usage row with idempotency.

        Uses DB-level UNIQUE(tenant_id, request_id) to prevent double-log.
        On IntegrityError (duplicate) -> silently ignore, return False.
        Never raises — caller is fail-open.
        """
        usage = UsageLedger(
            user_id=user_id,
            api_key_id=api_key_id,
            tenant_id=tenant_id,
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tokens_total=tokens_total,
            file_size_bytes=file_size_bytes,
            request_cost=request_cost,
            status_code=status_code,
            success=success,
        )
        try:
            db.add(usage)
            await db.commit()
            return True
        except IntegrityError:
            # Duplicate (tenant_id, request_id) — already logged, ignore.
            await db.rollback()
            logger.debug("usage.duplicate_skipped tenant_id=%s request_id=%s", tenant_id, request_id)
            return False
        except Exception:
            await db.rollback()
            logger.error("usage.insert_failed tenant_id=%s request_id=%s", tenant_id, request_id, exc_info=True)
            return False
