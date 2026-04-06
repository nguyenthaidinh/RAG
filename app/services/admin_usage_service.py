from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models.usage import UsageLedger
from app.db.models.user import User
from datetime import datetime, timedelta, timezone


class AdminUsageService:
    @staticmethod
    async def get_usage_summary(
        db: AsyncSession,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        tenant_id: str | None = None,
    ) -> dict:
        """
        Get usage summary with totals and top users/endpoints.
        If tenant_id is provided, filter by tenant.
        """
        if not from_date:
            from_date = datetime.now(timezone.utc) - timedelta(days=30)
        if not to_date:
            to_date = datetime.now(timezone.utc)

        # Build base filters
        base_filters = [
            UsageLedger.created_at >= from_date,
            UsageLedger.created_at <= to_date,
        ]
        if tenant_id:
            base_filters.append(UsageLedger.tenant_id == tenant_id)

        # Total requests, tokens
        total_res = await db.execute(
            select(
                func.count(UsageLedger.id).label("total_requests"),
                func.sum(UsageLedger.tokens_total).label("total_tokens"),
                func.sum(UsageLedger.file_size_bytes).label("total_storage"),
            )
            .where(*base_filters)
        )
        total_row = total_res.first()

        # Top users by requests
        user_query = (
            select(
                User.id,
                User.email,
                func.count(UsageLedger.id).label("request_count"),
                func.sum(UsageLedger.tokens_total).label("tokens_total"),
            )
            .join(UsageLedger, User.id == UsageLedger.user_id)
            .where(*base_filters)
            .group_by(User.id, User.email)
            .order_by(desc(func.count(UsageLedger.id)))
            .limit(10)
        )
        top_users_res = await db.execute(user_query)
        top_users = [
            {
                "user_id": row.id,
                "email": row.email,
                "request_count": row.request_count or 0,
                "tokens_total": int(row.tokens_total or 0),
            }
            for row in top_users_res.all()
        ]

        # Top endpoints
        endpoint_query = (
            select(
                UsageLedger.endpoint,
                UsageLedger.method,
                func.count(UsageLedger.id).label("request_count"),
                func.sum(UsageLedger.tokens_total).label("tokens_total"),
            )
            .where(*base_filters)
            .group_by(UsageLedger.endpoint, UsageLedger.method)
            .order_by(desc(func.count(UsageLedger.id)))
            .limit(10)
        )
        top_endpoints_res = await db.execute(endpoint_query)
        top_endpoints = [
            {
                "endpoint": row.endpoint,
                "method": row.method,
                "request_count": row.request_count or 0,
                "tokens_total": int(row.tokens_total or 0),
            }
            for row in top_endpoints_res.all()
        ]

        return {
            "total_requests": total_row.total_requests or 0,
            "total_tokens": int(total_row.total_tokens or 0),
            "total_storage": int(total_row.total_storage or 0),
            "top_users": top_users,
            "top_endpoints": top_endpoints,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
        }

    @staticmethod
    async def get_user_usage(
        db: AsyncSession,
        user_id: int,
    ) -> dict:
        """
        Get user usage with daily stats and latest logs.
        """
        # Daily stats (last 30 days)
        since = datetime.now(timezone.utc) - timedelta(days=30)

        daily_stats_res = await db.execute(
            select(
                func.date(UsageLedger.created_at).label("date"),
                func.count(UsageLedger.id).label("request_count"),
                func.sum(UsageLedger.tokens_total).label("tokens_total"),
            )
            .where(
                UsageLedger.user_id == user_id,
                UsageLedger.created_at >= since,
            )
            .group_by(func.date(UsageLedger.created_at))
            .order_by(desc(func.date(UsageLedger.created_at)))
        )
        daily_stats = [
            {
                "date": row.date.isoformat() if row.date else None,
                "request_count": row.request_count or 0,
                "tokens_total": int(row.tokens_total or 0),
            }
            for row in daily_stats_res.all()
        ]

        # Latest 50 logs
        latest_logs_res = await db.execute(
            select(UsageLedger)
            .where(UsageLedger.user_id == user_id)
            .order_by(UsageLedger.created_at.desc())
            .limit(50)
        )
        latest_logs = list(latest_logs_res.scalars().all())

        return {
            "daily_stats": daily_stats,
            "latest_logs": latest_logs,
        }
