"""
Phase 1.2A — Programmatic (deterministic) graders.

All graders here are **pure functions** — no LLM, no network calls.
They take structured inputs and return deterministic outputs.

Organization:
  Retrieval graders:  evaluate document retrieval quality.
  Answer graders:     evaluate generated answer quality.

Fail-safe rules:
  - Empty/None inputs -> safe default (False / 0.0).
  - No grader ever raises an exception — wrap in try/except with
    a clear failure reason if something goes wrong.
"""
from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL GRADERS
# ═══════════════════════════════════════════════════════════════════════════════


def retrieved_expected_doc(
    result_doc_ids: list[int],
    expected_doc_ids: list[int],
) -> tuple[bool, int]:
    """Check if any expected document was retrieved.

    Returns:
        (any_hit, hit_count): whether at least one expected doc was
        found, and how many were found.
    """
    if not expected_doc_ids or not result_doc_ids:
        return (False, 0)

    expected = set(expected_doc_ids)
    found = expected & set(result_doc_ids)
    return (len(found) > 0, len(found))


def hit_at_k(
    result_doc_ids: list[int],
    expected_doc_ids: list[int],
    k: int,
) -> bool:
    """Check if any expected document appears in the top-k results.

    Args:
        result_doc_ids: Ranked list of retrieved document IDs (best first).
        expected_doc_ids: Document IDs that should be present.
        k: Cutoff rank.

    Returns:
        True if at least one expected doc is in result_doc_ids[:k].
    """
    if not expected_doc_ids or not result_doc_ids or k <= 0:
        return False

    return bool(set(expected_doc_ids) & set(result_doc_ids[:k]))


def recall_at_k(
    result_doc_ids: list[int],
    expected_doc_ids: list[int],
    k: int,
) -> float:
    """Fraction of expected docs found in top-k results.

    Returns:
        Float in [0.0, 1.0]. 0.0 if expected_doc_ids is empty.
    """
    if not expected_doc_ids or k <= 0:
        return 0.0
    if not result_doc_ids:
        return 0.0

    expected = set(expected_doc_ids)
    found = expected & set(result_doc_ids[:k])
    return len(found) / len(expected)


def mrr(
    result_doc_ids: list[int],
    expected_doc_ids: list[int],
) -> float:
    """Mean Reciprocal Rank — reciprocal of the rank of the first hit.

    Returns:
        Float in (0.0, 1.0] if a hit is found; 0.0 otherwise.
    """
    if not expected_doc_ids or not result_doc_ids:
        return 0.0

    expected = set(expected_doc_ids)
    for rank, doc_id in enumerate(result_doc_ids, start=1):
        if doc_id in expected:
            return 1.0 / rank

    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER GRADERS
# ═══════════════════════════════════════════════════════════════════════════════


def has_answer(answer_text: str | None) -> bool:
    """Check if the answer contains meaningful text (>20 chars)."""
    if not answer_text:
        return False
    return len(answer_text.strip()) > 20


def has_citations(answer_result: dict | None) -> bool:
    """Check if the answer result contains any citations.

    Args:
        answer_result: Full response dict. Expected keys:
            - "citations": list[dict] — citation objects.
    """
    if not answer_result:
        return False
    citations = answer_result.get("citations", [])
    return isinstance(citations, list) and len(citations) > 0


def citation_doc_ids_exist(answer_result: dict | None) -> bool:
    """Check that all citations have a non-null document_id.

    Returns False if there are no citations at all.
    """
    if not answer_result:
        return False
    citations = answer_result.get("citations", [])
    if not citations:
        return False
    return all(
        c.get("document_id") is not None or c.get("source_document_id") is not None
        for c in citations
    )


def citations_same_tenant(
    answer_result: dict | None,
    tenant_id: str,
    *,
    tenant_doc_ids: set[int] | None = None,
) -> bool:
    """Check that citations don't reference documents from other tenants.

    If tenant_doc_ids is provided, verify every cited document_id is in
    that set. Otherwise, this is a structural check only (all citations
    must have a valid document_id).

    In 1.2A without a live DB lookup, this defaults to structural check.
    """
    if not answer_result:
        return True  # No citations -> no cross-tenant leak

    citations = answer_result.get("citations", [])
    if not citations:
        return True

    if tenant_doc_ids is not None:
        for c in citations:
            doc_id = c.get("document_id") or c.get("source_document_id")
            if doc_id is not None and doc_id not in tenant_doc_ids:
                return False

    # Structural check: all citations must have doc IDs
    return citation_doc_ids_exist(answer_result)


def keyword_coverage(
    answer_text: str | None,
    expected_keywords: list[str],
) -> tuple[float, list[str], list[str]]:
    """Fraction of expected keywords found in the answer text.

    Returns:
        (coverage_ratio, found_keywords, missing_keywords)
    """
    if not expected_keywords:
        return (1.0, [], [])  # No expectations -> trivially satisfied
    if not answer_text:
        return (0.0, [], list(expected_keywords))

    answer_lower = answer_text.lower()
    found = [k for k in expected_keywords if k.lower() in answer_lower]
    missing = [k for k in expected_keywords if k.lower() not in answer_lower]

    coverage = len(found) / len(expected_keywords)
    return (coverage, found, missing)


def forbidden_keyword_violation(
    answer_text: str | None,
    forbidden_keywords: list[str],
) -> tuple[bool, list[str]]:
    """Check if the answer contains any forbidden keywords.

    Returns:
        (has_violation, violating_keywords)
    """
    if not forbidden_keywords or not answer_text:
        return (False, [])

    answer_lower = answer_text.lower()
    violations = [k for k in forbidden_keywords if k.lower() in answer_lower]
    return (len(violations) > 0, violations)


def abstention_detected(answer_text: str | None) -> bool:
    """Check if the answer contains abstention/no-answer indicators.

    Checks both Vietnamese and English indicators.
    """
    if not answer_text:
        return True  # No text at all -> effectively abstaining

    text_lower = answer_text.lower()

    indicators = (
        # Vietnamese
        "không tìm thấy",
        "không có thông tin",
        "không thể trả lời",
        "không có dữ liệu",
        "không tìm được",
        "không có tài liệu",
        # English
        "could not find",
        "no relevant",
        "i don't have",
        "no information",
        "cannot answer",
        "i couldn't find",
        "unable to find",
        "no data available",
    )

    return any(ind in text_lower for ind in indicators)


def abstention_behavior_basic(
    answer_text: str | None,
    expected_answer_type: str,
) -> bool:
    """Check if the abstention behavior is correct for the expected type.

    Rules:
      - "abstain" / "no_answer": should abstain (detect abstention indicator).
      - Other types: should NOT abstain (should provide a real answer).

    Returns:
        True if behavior matches expectation.
    """
    should_abstain = expected_answer_type in ("abstain", "no_answer")
    did_abstain = abstention_detected(answer_text)

    if should_abstain:
        return did_abstain
    else:
        # For non-abstain types, we want a real answer
        return not did_abstain
