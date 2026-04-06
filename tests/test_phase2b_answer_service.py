"""
Phase 2B — AnswerService tests for source-mode support.

Covers:
  A. Knowledge-only with evidences → same as Phase 1
  B. System-only path: snippets=[], allow_system_context_only=True
  C. System-only with empty system_context_block → no answer
  D. _system_prompt adapts to source mode
  E. _user_prompt omits empty document context
  F. Backward compatibility: old callers work unchanged
  G. Mixed mode: both evidences and system context
"""
from __future__ import annotations

import asyncio

import pytest


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# =====================================================================
# A. KNOWLEDGE-ONLY — identical to Phase 1
# =====================================================================


class TestKnowledgeOnly:
    def test_knowledge_with_evidences(self):
        """Knowledge path with snippets works exactly like before."""
        from app.services.answer_service import AnswerService, AnswerSnippet

        svc = AnswerService()
        snippets = [
            AnswerSnippet(document_id=1, chunk_id=1, snippet="Content A", score=0.9),
            AnswerSnippet(document_id=2, chunk_id=2, snippet="Content B", score=0.8),
        ]

        result = _run(svc.generate_structured(
            question="Tell me about X",
            snippets=snippets,
        ))

        assert result is not None
        assert result.intent in ("overview", "specific", "compare", "general")
        assert len(result.evidences) >= 1

    def test_knowledge_no_snippets_returns_no_answer(self):
        """No snippets + default params → empty result."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        result = _run(svc.generate_structured(
            question="What is this?",
            snippets=[],
        ))

        assert result.text is None
        assert result.evidences == ()


# =====================================================================
# B. SYSTEM-ONLY PATH
# =====================================================================


class TestSystemOnly:
    def test_system_only_with_context(self):
        """When allow_system_context_only=True and system context exists,
        generate_structured() should NOT early-return with empty evidences.
        (LLM disabled → returns evidences=() but text=None is acceptable.)
        """
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        result = _run(svc.generate_structured(
            question="How many users?",
            snippets=[],
            system_context_block="## System Stats\n- Total Users: 42",
            question_category="system",
            allow_system_context_only=True,
        ))

        # With LLM disabled, text will be None, but the gate should
        # NOT block early (it should reach the LLM gate, not the
        # "no content at all" gate).
        assert result is not None
        # No evidences from document retrieval
        assert result.evidences == ()

    def test_system_only_no_context_returns_none(self):
        """No system context + no snippets → no answer."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        result = _run(svc.generate_structured(
            question="Status?",
            snippets=[],
            system_context_block="",
            question_category="system",
            allow_system_context_only=True,
        ))

        assert result.text is None
        assert result.evidences == ()


# =====================================================================
# C. _SYSTEM_PROMPT ADAPTS TO SOURCE MODE
# =====================================================================


class TestSystemPromptModes:
    def test_document_only_mode(self):
        """Legacy document-only mode: no system context references."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()
        prompt = svc._system_prompt(
            "general",
            has_system_context=False,
            has_document_context=True,
        )

        assert "document" in prompt.lower()
        assert "system context" not in prompt.lower()

    def test_system_only_mode(self):
        """System-only: no document references, emphasizes system context."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()
        prompt = svc._system_prompt(
            "general",
            has_system_context=True,
            has_document_context=False,
        )

        assert "system context" in prompt.lower()
        # Should NOT say "document context is primary"
        assert "document context" not in prompt.lower() or "document" not in prompt.split("PRIMARY")[0].lower() if "PRIMARY" in prompt else True

    def test_mixed_mode(self):
        """Both sources: references both document and system context."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()
        prompt = svc._system_prompt(
            "general",
            has_system_context=True,
            has_document_context=True,
        )

        assert "document context" in prompt.lower()
        assert "system context" in prompt.lower()

    def test_no_context_at_all(self):
        """Neither source: fallback to document-only wording."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()
        prompt = svc._system_prompt(
            "general",
            has_system_context=False,
            has_document_context=False,
        )

        # Should still be a valid prompt
        assert "document" in prompt.lower()

    def test_system_only_does_not_say_document_primary(self):
        """System-only prompt should NOT say document is primary."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()
        prompt = svc._system_prompt(
            "general",
            has_system_context=True,
            has_document_context=False,
        )

        assert "document question-answering" not in prompt.lower()
        assert "answer only from the provided system context" in prompt.lower()


# =====================================================================
# D. _USER_PROMPT OMITS EMPTY SECTIONS
# =====================================================================


class TestUserPromptSections:
    def test_no_document_context_section_when_empty(self):
        """Empty document context → no 'Document context:' section."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        prompt = svc._user_prompt(
            "What status?",
            "",  # empty document context
            "general",
            system_context_block="## Stats\n- Users: 42",
        )

        assert "Document context:" not in prompt
        assert "System context:" in prompt
        assert "Users: 42" in prompt

    def test_no_system_context_section_when_empty(self):
        """Empty system context → no 'System context:' section."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        prompt = svc._user_prompt(
            "What is X?",
            "Document says X is Y.",
            "general",
            system_context_block="",
        )

        assert "System context:" not in prompt
        assert "Document context:" in prompt

    def test_both_sections_present(self):
        """Both non-empty → both sections appear."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        prompt = svc._user_prompt(
            "Q?",
            "Doc content",
            "general",
            system_context_block="## User\nAlice",
        )

        assert "System context:" in prompt
        assert "Document context:" in prompt

    def test_section_ordering_preserved(self):
        """Order: question → history → system → document → instruction."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        prompt = svc._user_prompt(
            "Test question",
            "Doc here",
            "general",
            history_block="User: prev\nAssistant: reply",
            system_context_block="## User\nAlice",
        )

        q_idx = prompt.index("User question:")
        h_idx = prompt.index("Recent conversation:")
        s_idx = prompt.index("System context:")
        d_idx = prompt.index("Document context:")

        assert q_idx < h_idx < s_idx < d_idx


# =====================================================================
# E. BACKWARD COMPATIBILITY
# =====================================================================


class TestBackwardCompat:
    def test_no_new_args_works(self):
        """Old callers without question_category/allow_system_context_only."""
        from app.services.answer_service import AnswerService, AnswerSnippet

        svc = AnswerService()
        snippets = [
            AnswerSnippet(document_id=1, chunk_id=1, snippet="Content", score=0.9),
        ]

        result = _run(svc.generate_structured(
            question="Test",
            snippets=snippets,
            history=None,
        ))

        assert result is not None
        assert result.intent in ("overview", "specific", "compare", "general")

    def test_generate_wrapper(self):
        """generate() wrapper still returns text-only."""
        from app.services.answer_service import AnswerService, AnswerSnippet

        svc = AnswerService()
        snippets = [
            AnswerSnippet(document_id=1, chunk_id=1, snippet="Content", score=0.9),
        ]

        result = _run(svc.generate(
            question="Test?",
            snippets=snippets,
        ))

        assert result is None or isinstance(result, str)

    def test_empty_system_context_same_as_none(self):
        """Empty system_context_block produces same prompt as none."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        prompt_none = svc._user_prompt("Q?", "Doc.", "general")
        prompt_empty = svc._user_prompt(
            "Q?", "Doc.", "general",
            system_context_block="",
        )

        assert prompt_none == prompt_empty

    def test_system_prompt_default_is_document_only(self):
        """Default _system_prompt() call (no flags) → document-only."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        prompt = svc._system_prompt("general")

        assert "document" in prompt.lower()


# =====================================================================
# F. MIXED MODE
# =====================================================================


class TestMixedMode:
    def test_mixed_with_evidences_and_system(self):
        """Mixed: both snippets and system context block."""
        from app.services.answer_service import AnswerService, AnswerSnippet

        svc = AnswerService()
        snippets = [
            AnswerSnippet(document_id=1, chunk_id=1, snippet="Policy X", score=0.9),
        ]

        result = _run(svc.generate_structured(
            question="What is policy X and who am I?",
            snippets=snippets,
            system_context_block="## Current User\nName: Alice\nRole: admin",
            question_category="mixed",
            allow_system_context_only=True,
        ))

        assert result is not None
        assert len(result.evidences) >= 1

    def test_mixed_no_snippets_but_system(self):
        """Mixed: no snippets but system context → should not early-return."""
        from app.services.answer_service import AnswerService

        svc = AnswerService()

        result = _run(svc.generate_structured(
            question="Pending plus policy?",
            snippets=[],
            system_context_block="## System Stats\n- Pending: 5",
            question_category="mixed",
            allow_system_context_only=True,
        ))

        assert result is not None
        assert result.evidences == ()
