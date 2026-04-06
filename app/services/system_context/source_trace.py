"""
Internal answer-source provenance trace (Phase 2C).

A frozen dataclass that captures which sources contributed to
a particular assistant response.  Used ONLY for:
  - structured logging
  - debug endpoint
  - eval / staging tests

NOT part of the public API — never returned to external callers.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnswerSourceTrace:
    """Provenance trace for a single assistant response."""

    # Orchestration routing
    category: str = "knowledge"
    orchestration_ok: bool = False

    # Source routing flags
    should_use_knowledge: bool = True
    should_use_system_context: bool = False
    should_use_access_context: bool = False

    # What actually happened
    retrieval_skipped: bool = False
    used_document_evidence: bool = False
    used_system_context: bool = False

    # Evidence counts
    snippets_count: int = 0
    evidences_count: int = 0
    system_context_chars: int = 0

    # Answer source
    answer_source: str = "deterministic"  # llm | system_fallback | evidence_fallback | deterministic
    fallback_level: int = 0  # 0=llm, 1=evidence_fallback, 2=system_fallback, 3=deterministic

    # LLM metadata
    intent: str = "general"
    llm_provider: str | None = None
    llm_model: str | None = None
    used_history: bool = False

    def to_log_dict(self) -> dict:
        """Return a safe dict for structured logging.

        No raw content, no PII, no tokens — only routing metadata.
        """
        return {
            "category": self.category,
            "orchestration_ok": self.orchestration_ok,
            "should_use_knowledge": self.should_use_knowledge,
            "should_use_system_context": self.should_use_system_context,
            "should_use_access_context": self.should_use_access_context,
            "retrieval_skipped": self.retrieval_skipped,
            "used_document_evidence": self.used_document_evidence,
            "used_system_context": self.used_system_context,
            "snippets_count": self.snippets_count,
            "evidences_count": self.evidences_count,
            "system_context_chars": self.system_context_chars,
            "answer_source": self.answer_source,
            "fallback_level": self.fallback_level,
            "intent": self.intent,
            "used_history": self.used_history,
        }
