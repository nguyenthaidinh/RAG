# from __future__ import annotations

# from typing import Sequence

# from sqlalchemy import Select, and_, cast, func, literal, or_, select, String
# from sqlalchemy.ext.asyncio import AsyncSession

# from app.db.models.document import Document
# from app.schemas.metadata_search import MetadataSearchConditions


# class DocumentMetadataRepo:
#     async def search_candidates(
#         self,
#         db: AsyncSession,
#         *,
#         tenant_id: str,
#         conditions: MetadataSearchConditions,
#         limit: int = 20,
#     ) -> Sequence[Document]:
#         stmt: Select = (
#             select(Document)
#             .where(Document.tenant_id == tenant_id)
#             .where(Document.status == "ready")
#         )

#         filters = []

#         # document_kind
#         if conditions.document_kinds:
#             filters.append(
#                 cast(Document.meta["document_identity"]["document_kind"].astext, String).in_(
#                     conditions.document_kinds
#                 )
#             )

#         # source_label
#         if conditions.source_labels:
#             filters.append(
#                 cast(Document.meta["document_identity"]["source_label"].astext, String).in_(
#                     conditions.source_labels
#                 )
#             )

#         # language
#         if conditions.languages:
#             filters.append(
#                 cast(Document.meta["document_identity"]["language"].astext, String).in_(
#                     conditions.languages
#                 )
#             )

#         # title terms
#         for term in conditions.title_terms:
#             filters.append(
#                 func.lower(Document.title).like(f"%{term.lower()}%")
#             )

#         # topics: JSON array string contains
#         topic_filters = []
#         for topic in conditions.topics:
#             topic_filters.append(
#                 cast(Document.meta["semantic"]["topics"], String).ilike(f"%{topic}%")
#             )
#         if topic_filters:
#             filters.append(or_(*topic_filters))

#         # tags
#         tag_filters = []
#         for tag in conditions.tags:
#             tag_filters.append(
#                 cast(Document.meta["semantic"]["tags"], String).ilike(f"%{tag}%")
#             )
#         if tag_filters:
#             filters.append(or_(*tag_filters))

#         # audience
#         audience_filters = []
#         for aud in conditions.audience:
#             audience_filters.append(
#                 cast(Document.meta["document_identity"]["audience"], String).ilike(f"%{aud}%")
#             )
#         if audience_filters:
#             filters.append(or_(*audience_filters))

#         # keywords
#         keyword_filters = []
#         for kw in conditions.keywords:
#             keyword_filters.append(
#                 or_(
#                     func.lower(Document.title).like(f"%{kw.lower()}%"),
#                     cast(Document.meta["semantic"]["keywords"], String).ilike(f"%{kw}%"),
#                     cast(Document.meta["retrieval_hints"]["search_boost_terms"], String).ilike(f"%{kw}%"),
#                 )
#             )
#         if keyword_filters:
#             filters.append(or_(*keyword_filters))

#         if filters:
#             stmt = stmt.where(and_(*filters))

#         stmt = stmt.order_by(Document.id.desc()).limit(limit)
#         result = await db.execute(stmt)
#         return result.scalars().all()

from __future__ import annotations

from typing import Sequence

from sqlalchemy import Select, and_, cast, func, or_, select, String
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.schemas.metadata_search import MetadataSearchConditions


class DocumentMetadataRepo:
    async def search_candidates(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        conditions: MetadataSearchConditions,
        limit: int = 20,
    ) -> Sequence[Document]:
        stmt: Select = (
            select(Document)
            .where(Document.tenant_id == tenant_id)
            .where(Document.status == "ready")
        )

        filters = []

        # document_kind
        if conditions.document_kinds:
            filters.append(
                cast(Document.meta["document_identity"]["document_kind"].astext, String).in_(
                    conditions.document_kinds
                )
            )

        # source_label
        if conditions.source_labels:
            filters.append(
                cast(Document.meta["document_identity"]["source_label"].astext, String).in_(
                    conditions.source_labels
                )
            )

        # language
        if conditions.languages:
            filters.append(
                cast(Document.meta["document_identity"]["language"].astext, String).in_(
                    conditions.languages
                )
            )

        # title terms -> OR trong cùng nhóm
        title_filters = []
        for term in conditions.title_terms:
            title_filters.append(
                func.lower(Document.title).like(f"%{term.lower()}%")
            )
        if title_filters:
            filters.append(or_(*title_filters))

        # topics
        topic_filters = []
        for topic in conditions.topics:
            topic_filters.append(
                cast(Document.meta["semantic"]["topics"], String).ilike(f"%{topic}%")
            )
        if topic_filters:
            filters.append(or_(*topic_filters))

        # tags
        tag_filters = []
        for tag in conditions.tags:
            tag_filters.append(
                cast(Document.meta["semantic"]["tags"], String).ilike(f"%{tag}%")
            )
        if tag_filters:
            filters.append(or_(*tag_filters))

        # audience
        audience_filters = []
        for aud in conditions.audience:
            audience_filters.append(
                cast(Document.meta["document_identity"]["audience"], String).ilike(f"%{aud}%")
            )
        if audience_filters:
            filters.append(or_(*audience_filters))

        # keywords
        keyword_filters = []
        for kw in conditions.keywords:
            keyword_filters.append(
                or_(
                    func.lower(Document.title).like(f"%{kw.lower()}%"),
                    cast(Document.meta["semantic"]["keywords"], String).ilike(f"%{kw}%"),
                    cast(Document.meta["retrieval_hints"]["search_boost_terms"], String).ilike(f"%{kw}%"),
                )
            )
        if keyword_filters:
            filters.append(or_(*keyword_filters))

        if filters:
            stmt = stmt.where(and_(*filters))

        stmt = stmt.order_by(Document.id.desc()).limit(limit)
        result = await db.execute(stmt)
        return result.scalars().all()