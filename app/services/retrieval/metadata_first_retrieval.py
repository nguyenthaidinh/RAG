from __future__ import annotations

import re
from typing import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.repos.document_metadata_repo import DocumentMetadataRepo
from app.schemas.metadata_search import MetadataCandidate, MetadataSearchConditions


class MetadataFirstRetrievalService:
    def __init__(self, repo: DocumentMetadataRepo | None = None):
        self.repo = repo or DocumentMetadataRepo()

    async def retrieve(
        self,
        db: AsyncSession,
        *,
        tenant_id: str,
        query: str,
        limit: int = 10,
    ) -> list[MetadataCandidate]:
        conditions = self._parse_query_to_conditions(query)
        docs = await self.repo.search_candidates(
            db,
            tenant_id=tenant_id,
            conditions=conditions,
            limit=limit,
        )
        candidates = self._score_candidates(query=query, docs=docs, conditions=conditions)
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def is_good_enough(self, candidates: list[MetadataCandidate]) -> bool:
        if not candidates:
            return False
        if len(candidates) >= 3 and candidates[0].score >= 0.60:
            return True
        if candidates[0].score >= 0.75:
            return True
        return False

    def _parse_query_to_conditions(self, query: str) -> MetadataSearchConditions:
        q = query.lower().strip()

        document_kinds: list[str] = []
        topics: list[str] = []
        tags: list[str] = []
        source_labels: list[str] = []
        languages: list[str] = []
        audience: list[str] = []
        title_terms: list[str] = []
        keywords: list[str] = []
        freshness_sensitive = False

        # document kind
        if any(x in q for x in ["quy chế", "quy định", "regulation", "policy"]):
            document_kinds.append("regulation")
            source_labels.append("academic_policy")
            freshness_sensitive = True

        if any(x in q for x in ["hướng dẫn", "guide", "instruction"]):
            document_kinds.append("guide")

        if any(x in q for x in ["thông báo", "notice", "announcement"]):
            document_kinds.append("notice")
            freshness_sensitive = True

        if any(x in q for x in ["mẫu đơn", "biểu mẫu", "form"]):
            document_kinds.append("form")

        if any(x in q for x in ["báo cáo", "report"]):
            document_kinds.append("report")

        # audience
        if "sinh viên" in q or "student" in q:
            audience.append("student")
        if "giảng viên" in q or "giáo viên" in q or "teacher" in q:
            audience.append("teacher")

        # topics
        topic_map = {
            "đào tạo": ["đào tạo", "tín chỉ", "học phần", "curriculum"],
            "tuyển sinh": ["tuyển sinh", "nhập học", "admission"],
            "học phí": ["học phí", "tuition"],
            "ai": ["ai", "trí tuệ nhân tạo", "machine learning"],
            "vật lí": ["vật lí", "physics"],
        }
        for topic, patterns in topic_map.items():
            if any(p in q for p in patterns):
                topics.append(topic)

        # language
        if re.search(r"\benglish\b|\btiếng anh\b", q):
            languages.append("en")
        if re.search(r"\bviệt\b|\btiếng việt\b", q):
            languages.append("vi")

        # title terms / keywords
        raw_terms = re.findall(r"[\wÀ-ỹ]{3,}", q)
        stop = {"cho", "của", "với", "đang", "cần", "những", "các", "trong", "văn", "bản"}
        cleaned_terms = [t for t in raw_terms if t not in stop]

        title_terms.extend(cleaned_terms[:5])
        keywords.extend(cleaned_terms[:8])
        tags.extend(topics)

        return MetadataSearchConditions(
            document_kinds=list(dict.fromkeys(document_kinds)),
            topics=list(dict.fromkeys(topics)),
            tags=list(dict.fromkeys(tags)),
            source_labels=list(dict.fromkeys(source_labels)),
            languages=list(dict.fromkeys(languages)),
            audience=list(dict.fromkeys(audience)),
            title_terms=list(dict.fromkeys(title_terms)),
            keywords=list(dict.fromkeys(keywords)),
            freshness_sensitive=freshness_sensitive,
        )

    def _score_candidates(
        self,
        *,
        query: str,
        docs: Sequence[Document],
        conditions: MetadataSearchConditions,
    ) -> list[MetadataCandidate]:
        q = query.lower()
        candidates: list[MetadataCandidate] = []

        for doc in docs:
            meta = doc.meta or {}
            identity = meta.get("document_identity") or {}
            semantic = meta.get("semantic") or {}
            hints = meta.get("retrieval_hints") or {}

            score = 0.0
            reasons: list[str] = []

            doc_kind = (identity.get("document_kind") or "").lower()
            doc_title = (doc.title or "").lower()
            doc_language = (identity.get("language") or "").lower()
            doc_topics = [str(x).lower() for x in (semantic.get("topics") or [])]
            doc_tags = [str(x).lower() for x in (semantic.get("tags") or [])]
            doc_keywords = [str(x).lower() for x in (semantic.get("keywords") or [])]
            doc_boost_terms = [str(x).lower() for x in (hints.get("search_boost_terms") or [])]
            doc_audience = [str(x).lower() for x in (identity.get("audience") or [])]
            doc_source_label = (identity.get("source_label") or "").lower()

            if conditions.document_kinds and doc_kind in conditions.document_kinds:
                score += 0.30
                reasons.append(f"document_kind={doc_kind}")

            if conditions.source_labels and doc_source_label in conditions.source_labels:
                score += 0.15
                reasons.append(f"source_label={doc_source_label}")

            if conditions.languages and doc_language in conditions.languages:
                score += 0.05
                reasons.append(f"language={doc_language}")

            topic_hits = set(conditions.topics).intersection(set(doc_topics))
            if topic_hits:
                score += min(0.20, 0.08 * len(topic_hits))
                reasons.append(f"topics={sorted(topic_hits)}")

            tag_hits = set(conditions.tags).intersection(set(doc_tags))
            if tag_hits:
                score += min(0.10, 0.05 * len(tag_hits))
                reasons.append(f"tags={sorted(tag_hits)}")

            audience_hits = set(conditions.audience).intersection(set(doc_audience))
            if audience_hits:
                score += 0.08
                reasons.append(f"audience={sorted(audience_hits)}")

            title_hits = [term for term in conditions.title_terms if term in doc_title]
            if title_hits:
                score += min(0.12, 0.04 * len(title_hits))
                reasons.append(f"title_terms={title_hits}")

            # kw_hits = [kw for kw in conditions.keywords if kw in doc_keywords or kw in doc_boost_terms or kw in q]
            kw_hits = [
                kw for kw in conditions.keywords
                if kw in doc_keywords or kw in doc_boost_terms
            ]
            if kw_hits:
                score += min(0.10, 0.03 * len(set(kw_hits)))
                reasons.append(f"keywords={sorted(set(kw_hits))}")

            candidates.append(
                MetadataCandidate(
                    document_id=doc.id,
                    title=doc.title,
                    source=doc.source,
                    external_id=doc.external_id,
                    representation_type=doc.representation_type,
                    parent_document_id=doc.parent_document_id,
                    metadata=meta,
                    content_text=doc.content_text,
                    score=round(score, 4),
                    reasons=reasons,
                )
            )

        return candidates