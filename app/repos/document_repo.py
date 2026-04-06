from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document


class DocumentRepo:
    async def get_by_key(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        source: str,
        external_id: str,
    ) -> Document | None:
        result = await db.execute(
            select(Document).where(
                Document.tenant_id == tenant_id,
                Document.source == source,
                Document.external_id == external_id,
            )
        )
        return result.scalar_one_or_none()

    @staticmethod
    def add(db: AsyncSession, doc: Document) -> None:
        db.add(doc)
