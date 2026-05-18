"""
Mapping Draft Contract — R6.3A DTO / enum / normalize helpers.

Defines the shared contract for mapping drafts (objective↔outcome,
course↔outcome, CLO↔PLO).  These are **draft** structures for Laravel
to render as an editable matrix; RAG never writes official mappings.

Guards:
  - No DB access.
  - No LLM calls.
  - No retrieval / context-pack.
  - No file processing.
  - No Program / ProgramVersion / ProgramVersionRevision imports.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# Enums / Constants
# ═══════════════════════════════════════════════════════════════════════


class MappingDraftType:
    OBJECTIVE_OUTCOME = "objective_outcome"
    COURSE_OUTCOME = "course_outcome"
    COURSE_LEARNING_OUTCOME_PROGRAM_OUTCOME = "course_learning_outcome_program_outcome"

    ALL = frozenset([
        OBJECTIVE_OUTCOME,
        COURSE_OUTCOME,
        COURSE_LEARNING_OUTCOME_PROGRAM_OUTCOME,
    ])


class MappingSourceType:
    EXTRACTED_FROM_CURRENT_CURRICULUM = "extracted_from_current_curriculum"
    GENERATED_FROM_DRAFT = "generated_from_draft"
    USER_EDITED = "user_edited"
    IMPORTED = "imported"
    UNKNOWN = "unknown"

    ALL = frozenset([
        EXTRACTED_FROM_CURRENT_CURRICULUM,
        GENERATED_FROM_DRAFT,
        USER_EDITED,
        IMPORTED,
        UNKNOWN,
    ])


class MappingConfidence:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    ALL = frozenset([LOW, MEDIUM, HIGH])


class MappingStatus:
    DRAFT = "draft"
    NEEDS_REVIEW = "needs_review"
    USER_CONFIRMED = "user_confirmed"

    ALL = frozenset([DRAFT, NEEDS_REVIEW, USER_CONFIRMED])


# Contribution levels: 0=none, 1=low, 2=medium, 3=high
CONTRIBUTION_LEVEL_LABELS: dict[int, str] = {
    0: "no_contribution",
    1: "low",
    2: "medium",
    3: "high",
}

_VALID_CONTRIBUTION_LEVELS = frozenset([0, 1, 2, 3])


# ═══════════════════════════════════════════════════════════════════════
# Normalize helpers
# ═══════════════════════════════════════════════════════════════════════


def normalize_contribution_level(value: Any) -> tuple[int, list[str]]:
    """Normalize a contribution level to 0–3.

    Returns (level, warnings).

    Rules:
      None / "" / blank → 0
      0/1/2/3 or "0"/"1"/"2"/"3" → int
      "X"/"x" → 1 with warning
      anything else → 0 with warning
    """
    warnings: list[str] = []

    if value is None:
        return 0, warnings

    if isinstance(value, bool):
        # bool is subclass of int in Python; treat as invalid
        warnings.append(f"Normalized invalid contribution_level '{value}' to 0.")
        return 0, warnings

    if isinstance(value, int):
        if value in _VALID_CONTRIBUTION_LEVELS:
            return value, warnings
        warnings.append(f"Normalized invalid contribution_level '{value}' to 0.")
        return 0, warnings

    if isinstance(value, float):
        iv = int(value)
        if iv in _VALID_CONTRIBUTION_LEVELS and float(iv) == value:
            return iv, warnings
        warnings.append(f"Normalized invalid contribution_level '{value}' to 0.")
        return 0, warnings

    s = str(value).strip()
    if not s:
        return 0, warnings

    if s in ("0", "1", "2", "3"):
        return int(s), warnings

    if s.upper() == "X":
        warnings.append("Mapped 'X' to low contribution level (1).")
        return 1, warnings

    warnings.append(f"Normalized invalid contribution_level '{value}' to 0.")
    return 0, warnings


def normalize_confidence(value: Any) -> str:
    """Normalize confidence to low/medium/high; invalid → medium."""
    if not value:
        return MappingConfidence.MEDIUM
    v = str(value).strip().lower()
    if v in MappingConfidence.ALL:
        return v
    return MappingConfidence.MEDIUM


def normalize_mapping_status(value: Any) -> str:
    """Normalize mapping status; invalid → draft."""
    if not value:
        return MappingStatus.DRAFT
    v = str(value).strip().lower()
    if v in MappingStatus.ALL:
        return v
    return MappingStatus.DRAFT


def normalize_source_type(value: Any) -> str:
    """Normalize source type; invalid → unknown."""
    if not value:
        return MappingSourceType.UNKNOWN
    v = str(value).strip().lower()
    if v in MappingSourceType.ALL:
        return v
    return MappingSourceType.UNKNOWN


def dedupe_warnings(warnings: list[str]) -> list[str]:
    """Deduplicate warnings preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for w in warnings:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Dataclasses — Source ref
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MappingSourceRef:
    """Source reference for a mapping row.

    If the mapping has no source (e.g. user-edited), source_refs should
    be an empty list.  Never fabricate sources.
    """
    ai_document_id: int | None = None
    external_file_id: str | None = None
    filename: str | None = None
    document_role: str | None = None
    chunk_id: int | None = None
    chunk_index: int | None = None
    score: float | None = None
    quote: str | None = None
    update_cycle_id: str | None = None
    program_code: str | None = None
    program_id: str | None = None


# ═══════════════════════════════════════════════════════════════════════
# Dataclasses — Mapping rows
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class ObjectiveOutcomeMappingRow:
    """Ma trận mục tiêu đào tạo ↔ CĐR chương trình."""
    objective_code: str | None = None
    objective_content: str | None = None
    outcome_code: str | None = None
    outcome_content: str | None = None
    contribution_level: int = 0
    rationale: str | None = None
    confidence: str = MappingConfidence.MEDIUM
    status: str = MappingStatus.DRAFT
    source_type: str = MappingSourceType.UNKNOWN
    source_refs: list[MappingSourceRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CourseOutcomeMappingRow:
    """Ma trận học phần ↔ CĐR chương trình."""
    course_code: str | None = None
    course_name: str | None = None
    knowledge_block: str | None = None
    semester: str | None = None
    credits: float | None = None
    outcome_code: str | None = None
    outcome_content: str | None = None
    contribution_level: int = 0
    rationale: str | None = None
    confidence: str = MappingConfidence.MEDIUM
    status: str = MappingStatus.DRAFT
    source_type: str = MappingSourceType.UNKNOWN
    source_refs: list[MappingSourceRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class CourseLearningOutcomeProgramOutcomeMappingRow:
    """Ma trận CĐR học phần ↔ CĐR chương trình."""
    course_code: str | None = None
    course_name: str | None = None
    course_outcome_code: str | None = None
    course_outcome_content: str | None = None
    program_outcome_code: str | None = None
    program_outcome_content: str | None = None
    contribution_level: int = 0
    rationale: str | None = None
    confidence: str = MappingConfidence.MEDIUM
    status: str = MappingStatus.DRAFT
    source_type: str = MappingSourceType.UNKNOWN
    source_refs: list[MappingSourceRef] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# Dataclass — Full payload
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MappingDraftPayload:
    """Full mapping draft payload for persistence / API response."""
    update_cycle_id: str = ""
    program_id: str | None = None
    program_code: str | None = None
    program_name: str | None = None
    draft_type: str = "mapping_draft"
    objective_outcome_rows: list[ObjectiveOutcomeMappingRow] = field(default_factory=list)
    course_outcome_rows: list[CourseOutcomeMappingRow] = field(default_factory=list)
    clo_program_outcome_rows: list[CourseLearningOutcomeProgramOutcomeMappingRow] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_summary: dict = field(default_factory=dict)

    def build_source_summary(self) -> dict:
        """Compute source_summary from current rows."""
        doc_ids: set[int] = set()
        source_types: set[str] = set()

        for row in self.objective_outcome_rows:
            source_types.add(row.source_type)
            for ref in row.source_refs:
                if ref.ai_document_id is not None:
                    doc_ids.add(ref.ai_document_id)

        for row in self.course_outcome_rows:
            source_types.add(row.source_type)
            for ref in row.source_refs:
                if ref.ai_document_id is not None:
                    doc_ids.add(ref.ai_document_id)

        for row in self.clo_program_outcome_rows:
            source_types.add(row.source_type)
            for ref in row.source_refs:
                if ref.ai_document_id is not None:
                    doc_ids.add(ref.ai_document_id)

        self.source_summary = {
            "documents_used": sorted(doc_ids),
            "rows_count": {
                MappingDraftType.OBJECTIVE_OUTCOME: len(self.objective_outcome_rows),
                MappingDraftType.COURSE_OUTCOME: len(self.course_outcome_rows),
                MappingDraftType.COURSE_LEARNING_OUTCOME_PROGRAM_OUTCOME: len(self.clo_program_outcome_rows),
            },
            "source_types": sorted(source_types),
        }
        return self.source_summary

    def rows_by_mapping_type(self) -> dict[str, list]:
        """Access rows keyed by canonical MappingDraftType constants.

        Convenience for Laravel-side rendering where the consumer
        iterates mapping types by canonical name.
        """
        return {
            MappingDraftType.OBJECTIVE_OUTCOME: self.objective_outcome_rows,
            MappingDraftType.COURSE_OUTCOME: self.course_outcome_rows,
            MappingDraftType.COURSE_LEARNING_OUTCOME_PROGRAM_OUTCOME: self.clo_program_outcome_rows,
        }

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return asdict(self)
