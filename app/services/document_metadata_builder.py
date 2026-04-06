from __future__ import annotations

import re
from collections import Counter
from typing import Any


class DocumentMetadataBuilder:
    STOPWORDS_VI = {
        "và", "là", "của", "cho", "trong", "với", "một", "các", "được",
        "theo", "tại", "về", "này", "khi", "đó", "đến", "từ", "có", "không"
    }

    def build(
        self,
        *,
        title: str | None,
        text: str,
        file_name: str | None,
        original_name: str | None,
        content_type: str | None,
        size_bytes: int | None,
        ingest_via: str,
    ) -> dict[str, Any]:
        clean_text = (text or "").strip()
        normalized_title = (title or original_name or file_name or "").strip()

        language = self._detect_language(clean_text)
        keywords = self._extract_keywords(clean_text, top_k=12)
        topics = self._infer_topics(normalized_title, clean_text, keywords)
        document_kind = self._infer_document_kind(normalized_title, clean_text, file_name)
        audience = self._infer_audience(clean_text)
        domain = self._infer_domain(clean_text, keywords)
        summary = self._make_summary(clean_text)
        entities = self._extract_entities(clean_text)
        structure = self._analyze_structure(clean_text)
        retrieval_hints = self._build_retrieval_hints(
            title=normalized_title,
            document_kind=document_kind,
            keywords=keywords,
            topics=topics,
        )

        return {
            "system": {
                "pipeline_mode": "semantic",
                "pipeline_version": "semantic_v1",
                "ingest_via": ingest_via,
                "file_name": file_name,
                "original_name": original_name,
                "content_type": content_type,
                "size_bytes": size_bytes,
            },
            "document_identity": {
                "title": normalized_title or None,
                "language": language,
                "document_kind": document_kind,
                "source_label": self._infer_source_label(document_kind),
                "audience": audience,
                "domain": domain,
            },
            "semantic": {
                "summary": summary,
                "keywords": keywords,
                "topics": topics,
                "tags": self._make_tags(document_kind, topics, audience),
                "entities": entities,
            },
            "structure": structure,
            "retrieval_hints": retrieval_hints,
        }

    def _detect_language(self, text: str) -> str:
        vi_chars = "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệóòỏõọốồổỗộớờởỡợúùủũụứừửữựíìỉĩịýỳỷỹỵ"
        if any(ch in text.lower() for ch in vi_chars):
            return "vi"
        return "unknown"

    def _extract_keywords(self, text: str, top_k: int = 10) -> list[str]:
        words = re.findall(r"\b[\wÀ-ỹ]{3,}\b", text.lower())
        words = [w for w in words if w not in self.STOPWORDS_VI and not w.isdigit()]
        freq = Counter(words)
        return [w for w, _ in freq.most_common(top_k)]

    def _infer_document_kind(self, title: str, text: str, file_name: str | None) -> str:
        haystack = f"{title}\n{text[:3000]}".lower()
        if any(k in haystack for k in ["quy chế", "quy định", "regulation", "policy"]):
            return "regulation"
        if any(k in haystack for k in ["hướng dẫn", "instruction", "guide"]):
            return "guide"
        if any(k in haystack for k in ["thông báo", "notice", "announcement"]):
            return "notice"
        if any(k in haystack for k in ["mẫu đơn", "biểu mẫu", "form"]):
            return "form"
        if any(k in haystack for k in ["báo cáo", "report"]):
            return "report"
        if any(k in haystack for k in ["đề thi", "câu hỏi", "bài tập"]):
            return "assessment"
        return "general"

    def _infer_topics(self, title: str, text: str, keywords: list[str]) -> list[str]:
        haystack = f"{title}\n{text[:5000]}".lower()
        topics = []

        mapping = {
            "tuyển sinh": ["tuyển sinh", "nhập học", "admission"],
            "đào tạo": ["đào tạo", "tín chỉ", "học phần", "curriculum"],
            "học phí": ["học phí", "miễn giảm", "tuition"],
            "khoa học": ["nghiên cứu", "khoa học", "scientific"],
            "vật lí": ["vật lí", "physics"],
            "ai": ["ai", "trí tuệ nhân tạo", "machine learning"],
        }

        for topic, patterns in mapping.items():
            if any(p in haystack for p in patterns):
                topics.append(topic)

        for kw in keywords[:5]:
            if kw not in topics:
                topics.append(kw)

        return topics[:10]

    def _infer_audience(self, text: str) -> list[str]:
        t = text.lower()
        audience = []
        if "học sinh" in t or "sinh viên" in t:
            audience.append("student")
        if "giáo viên" in t or "giảng viên" in t:
            audience.append("teacher")
        if "phụ huynh" in t:
            audience.append("parent")
        return audience or ["general"]

    def _infer_domain(self, text: str, keywords: list[str]) -> list[str]:
        t = text.lower()
        domain = []
        if any(k in t for k in ["trường", "học sinh", "giáo viên", "đào tạo"]):
            domain.append("education")
        if any(k in t for k in ["quy định", "quy chế", "thông báo"]):
            domain.append("policy")
        if any(k in t for k in ["ai", "machine learning", "mô hình"]):
            domain.append("technology")
        return domain or keywords[:3]

    def _make_summary(self, text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        joined = " ".join(lines[:5])
        return joined[:500] if joined else ""

    def _extract_entities(self, text: str) -> dict[str, list[str]]:
        dates = re.findall(r"\b(?:20\d{2}|19\d{2})\b", text)
        orgs = []
        for line in text.splitlines()[:50]:
            if "trường" in line.lower() or "university" in line.lower():
                orgs.append(line.strip()[:120])
        return {
            "organizations": list(dict.fromkeys(orgs))[:5],
            "people": [],
            "locations": [],
            "dates": list(dict.fromkeys(dates))[:10],
        }

    def _analyze_structure(self, text: str) -> dict[str, Any]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        heading_count = sum(1 for ln in lines if len(ln) < 120 and ln.endswith(":"))
        return {
            "has_headings": heading_count > 0,
            "estimated_sections": max(1, heading_count),
            "contains_tables": "|" in text or "\t" in text,
            "contains_forms": any(k in text.lower() for k in ["họ và tên", "mã số", "ký tên"]),
            "contains_questions": "?" in text or any(k in text.lower() for k in ["câu 1", "câu hỏi"]),
            "contains_procedures": any(k in text.lower() for k in ["bước 1", "quy trình", "thực hiện"]),
        }

    def _build_retrieval_hints(
        self,
        *,
        title: str,
        document_kind: str,
        keywords: list[str],
        topics: list[str],
    ) -> dict[str, Any]:
        preferred_representation = "original"
        if document_kind in {"guide", "report"}:
            preferred_representation = "synthesized"

        important_title_terms = re.findall(r"\b[\wÀ-ỹ]{3,}\b", title.lower())[:6]

        return {
            "preferred_representation": preferred_representation,
            "search_boost_terms": list(dict.fromkeys((important_title_terms + keywords[:5] + topics[:5])))[:12],
            "freshness_sensitive": document_kind in {"notice", "regulation"},
            "important_title_terms": important_title_terms,
        }

    def _infer_source_label(self, document_kind: str) -> str:
        mapping = {
            "regulation": "academic_policy",
            "guide": "instructional_doc",
            "notice": "announcement",
            "form": "template_form",
            "report": "reporting_doc",
            "assessment": "assessment_doc",
            "general": "general_doc",
        }
        return mapping.get(document_kind, "general_doc")

    def _make_tags(self, document_kind: str, topics: list[str], audience: list[str]) -> list[str]:
        tags = [document_kind] + topics[:5] + audience[:3]
        return list(dict.fromkeys(tags))[:12]