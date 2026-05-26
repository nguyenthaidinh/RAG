"""
Mapping AI Service — R6.13C1 + R6.13C1.1 orchestrator.

Orchestrate AI-powered objective↔outcome mapping suggestion:
  1. Normalize snapshots from request.
  2. Scoped retrieval via ctdt_retrieve() with MATRIX_MAPPING task type.
  3. Select/truncate context chunks for prompt budget.
  4. Call MappingAISkill (real LLM) with general_objective + used chunks.
  5. Validate + sort output.
  6. Compute coverage + quality gate.
  7. Return structured result with accurate retrieved/used chunk counts.

Guards:
  - Snapshot from Laravel is the ONLY source of M/C data.
  - Does NOT read draft DB as primary source.
  - Does NOT persist mapping draft.
  - Does NOT modify snapshot content.
  - Does NOT create M/C codes.
  - Does NOT change CĐR groups.
  - Does NOT fallback across update_cycle boundary.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ctdt_retrieval_service import (
    CTDTTaskType,
    ctdt_retrieve,
)

logger = logging.getLogger(__name__)


# ── Result DTO ───────────────────────────────────────────────────────


@dataclass
class MappingEntry:
    """A single validated mapping link."""
    objective_code: str
    outcome_code: str
    reason: str
    confidence: str = "medium"


@dataclass
class MappingCoverage:
    """Coverage summary."""
    objective_codes: list[str] = field(default_factory=list)
    outcome_codes: list[str] = field(default_factory=list)
    mapped_objective_codes: list[str] = field(default_factory=list)
    mapped_outcome_codes: list[str] = field(default_factory=list)
    unmapped_objective_codes: list[str] = field(default_factory=list)
    unmapped_outcome_codes: list[str] = field(default_factory=list)
    mapping_count: int = 0


@dataclass
class MappingSourceSummary:
    """RAG retrieval source metadata."""
    task_type: str = "matrix_mapping"
    retrieved_chunk_count: int = 0
    used_chunk_count: int = 0
    document_roles: list[str] = field(default_factory=list)
    latency_ms: int = 0


@dataclass
class MappingAIResult:
    """Full result from generate_mapping_ai."""
    status: str  # "generated" | "failed"
    update_cycle_id: str = ""
    program_id: str | None = None
    program_code: str | None = None
    program_name: str | None = None
    analysis_mode: str = "design"
    mappings: list[MappingEntry] = field(default_factory=list)
    coverage: MappingCoverage = field(default_factory=MappingCoverage)
    quality_level: str = "good"
    quality_messages: list[str] = field(default_factory=list)
    retrieval_used: bool = False
    source_summary: MappingSourceSummary = field(
        default_factory=MappingSourceSummary
    )
    warnings: list[str] = field(default_factory=list)


# ── Context budget constants ─────────────────────────────────────────

_MAX_CONTEXT_CHARS = 8000
_PER_CHUNK_LIMIT = 1500

# ── Retrieval query limits ───────────────────────────────────────────

_QUERY_CHAR_LIMIT = 1000


# ── Main orchestrator ────────────────────────────────────────────────


async def generate_mapping_ai(
    db: AsyncSession,
    *,
    tenant_id: str,
    user_id: int,
    update_cycle_id: str,
    program_id: str | None = None,
    program_code: str | None = None,
    program_name: str | None = None,
    analysis_mode: str = "design",
    general_objective: str | None = None,
    objectives: list[dict[str, str]],
    outcomes: list[dict[str, str]],
    top_k: int = 8,
    user_instruction: str | None = None,
    query_svc: Any = None,
) -> MappingAIResult:
    """
    Orchestrate AI mapping suggestion.

    Steps:
      1. Normalize code sets.
      2. Scoped retrieval with MATRIX_MAPPING task type.
      3. Select/truncate context chunks for prompt budget.
      4. Call MappingAISkill with general_objective + used chunks.
      5. Validate + sort output.
      6. Compute coverage + quality gate.
    """
    t0 = time.monotonic()

    # ── Step 1: Normalize code sets ──────────────────────────────
    objective_code_order = [o["code"].strip().upper() for o in objectives]
    outcome_code_order = [o["code"].strip().upper() for o in outcomes]
    all_obj_codes = set(objective_code_order)
    all_out_codes = set(outcome_code_order)

    # ── Step 2: Scoped retrieval ─────────────────────────────────
    retrieved_chunks: list[dict[str, Any]] = []

    retrieval_queries = _build_retrieval_queries(
        general_objective=general_objective,
        objectives=objectives,
        outcomes=outcomes,
        program_name=program_name,
    )

    retrieval_t0 = time.monotonic()
    retrieval_warnings: list[str] = []

    for query in retrieval_queries:
        try:
            result = await ctdt_retrieve(
                db,
                tenant_id=tenant_id,
                user_id=user_id,
                query=query,
                update_cycle_id=update_cycle_id,
                program_code=program_code,
                program_id=program_id,
                task_type=CTDTTaskType.MATRIX_MAPPING,
                document_roles=None,  # Use task policy roles
                top_k=top_k,
                query_svc=query_svc,
            )
            for ctx in result.contexts:
                retrieved_chunks.append({
                    "text": ctx.text,
                    "filename": ctx.filename,
                    "document_role": ctx.document_role,
                    "score": ctx.score,
                    "ai_document_id": ctx.ai_document_id,
                    "chunk_id": ctx.chunk_id,
                })
        except Exception:
            logger.exception(
                "mapping_ai.retrieval_failed update_cycle=%s query=%s",
                update_cycle_id, query[:80],
            )
            retrieval_warnings.append(
                "Lỗi truy xuất tài liệu bổ trợ; gợi ý sẽ dựa trên "
                "Mục tiêu và Chuẩn đầu ra đã hoàn thành."
            )

    # Deduplicate context chunks by (ai_document_id, chunk_id)
    retrieved_chunks = _dedupe_context_chunks(retrieved_chunks)

    retrieval_ms = int((time.monotonic() - retrieval_t0) * 1000)

    retrieved_chunk_count = len(retrieved_chunks)

    # ── Step 3: Select/truncate context for prompt budget ────────
    used_chunks = _select_context_chunks_for_prompt(
        retrieved_chunks,
        max_total_chars=_MAX_CONTEXT_CHARS,
        per_chunk_limit=_PER_CHUNK_LIMIT,
    )
    used_chunk_count = len(used_chunks)

    # Determine retrieval_used based on chunks actually sent to LLM
    if used_chunks:
        retrieval_used = True
        doc_roles_used = sorted({
            c.get("document_role", "")
            for c in used_chunks
            if c.get("document_role")
        })
    else:
        retrieval_used = False
        doc_roles_used = sorted({
            c.get("document_role", "")
            for c in retrieved_chunks
            if c.get("document_role")
        }) if retrieved_chunks else []
        if not retrieved_chunks:
            retrieval_warnings.append(
                "Không tìm thấy tài liệu bổ trợ trong phạm vi đợt cập nhật "
                "hiện tại; gợi ý được sinh dựa trên Mục tiêu và Chuẩn đầu ra "
                "đã hoàn thành."
            )
        elif used_chunk_count == 0:
            retrieval_warnings.append(
                "Tìm thấy tài liệu bổ trợ nhưng nội dung không đủ để đưa "
                "vào prompt; gợi ý dựa trên Mục tiêu và Chuẩn đầu ra đã hoàn thành."
            )

    source_summary = MappingSourceSummary(
        task_type="matrix_mapping",
        retrieved_chunk_count=retrieved_chunk_count,
        used_chunk_count=used_chunk_count,
        document_roles=doc_roles_used,
        latency_ms=retrieval_ms,
    )

    # ── Step 4: Call MappingAISkill ──────────────────────────────
    from app.services.ctdt_skills.mapping_ai_skill import (
        MappingAISkill,
        MappingAIStatus,
    )

    skill = MappingAISkill()

    skill_result = await skill.run(
        update_cycle_id=update_cycle_id,
        program_code=program_code,
        program_name=program_name,
        general_objective=general_objective,
        objectives=objectives,
        outcomes=outcomes,
        context_chunks=used_chunks,
        user_instruction=user_instruction,
    )

    # ── Step 5: Validate + sort output ───────────────────────────
    # Skill already validates codes and deduplicates, but we re-validate
    # as defense-in-depth.
    valid_mappings: list[MappingEntry] = []
    seen_pairs: set[tuple[str, str]] = set()
    all_warnings = list(retrieval_warnings)
    all_warnings.extend(skill_result.warnings)

    for candidate in skill_result.candidates:
        obj_code = candidate.objective_code.strip().upper()
        out_code = candidate.outcome_code.strip().upper()

        if obj_code not in all_obj_codes:
            all_warnings.append(
                f"Defense-in-depth: {obj_code} ngoài snapshot → bỏ qua."
            )
            continue
        if out_code not in all_out_codes:
            all_warnings.append(
                f"Defense-in-depth: {out_code} ngoài snapshot → bỏ qua."
            )
            continue

        pair = (obj_code, out_code)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        if not candidate.reason.strip():
            all_warnings.append(
                f"Defense-in-depth: {obj_code}↔{out_code} không có reason → bỏ qua."
            )
            continue

        valid_mappings.append(MappingEntry(
            objective_code=obj_code,
            outcome_code=out_code,
            reason=candidate.reason.strip(),
            confidence=candidate.confidence,
        ))

    # Sort: objective order → outcome order
    obj_order_map = {code: i for i, code in enumerate(objective_code_order)}
    out_order_map = {code: i for i, code in enumerate(outcome_code_order)}
    valid_mappings.sort(key=lambda m: (
        obj_order_map.get(m.objective_code, 999),
        out_order_map.get(m.outcome_code, 999),
    ))

    # ── Step 6: Coverage + quality gate ──────────────────────────
    mapped_obj = sorted({m.objective_code for m in valid_mappings},
                        key=lambda c: obj_order_map.get(c, 999))
    mapped_out = sorted({m.outcome_code for m in valid_mappings},
                        key=lambda c: out_order_map.get(c, 999))
    unmapped_obj = [c for c in objective_code_order if c not in mapped_obj]
    unmapped_out = [c for c in outcome_code_order if c not in mapped_out]

    coverage = MappingCoverage(
        objective_codes=objective_code_order,
        outcome_codes=outcome_code_order,
        mapped_objective_codes=mapped_obj,
        mapped_outcome_codes=mapped_out,
        unmapped_objective_codes=unmapped_obj,
        unmapped_outcome_codes=unmapped_out,
        mapping_count=len(valid_mappings),
    )

    # Quality gate
    quality_messages: list[str] = []

    if not valid_mappings:
        quality_level = "failed"
        quality_messages.append(
            "AI không đề xuất được liên kết hợp lệ nào giữa "
            "Mục tiêu đào tạo và Chuẩn đầu ra."
        )
    elif unmapped_obj or unmapped_out:
        quality_level = "warning"
        if unmapped_obj:
            quality_messages.append(
                f"Mục tiêu chưa có liên kết: {', '.join(unmapped_obj)}. "
                f"Vui lòng rà soát trước khi lưu."
            )
        if unmapped_out:
            quality_messages.append(
                f"Chuẩn đầu ra chưa có liên kết: {', '.join(unmapped_out)}. "
                f"Vui lòng rà soát trước khi lưu."
            )
    else:
        quality_level = "good"

    # Dense matrix warning
    max_possible = len(objective_code_order) * len(outcome_code_order)
    if max_possible > 0 and len(valid_mappings) > 0:
        density = len(valid_mappings) / max_possible
        if density > 0.8:
            quality_messages.append(
                f"Ma trận liên kết khá dày ({len(valid_mappings)}/{max_possible} ô). "
                f"Vui lòng kiểm tra và loại bỏ liên kết không thực sự cần thiết."
            )
            if quality_level == "good":
                quality_level = "warning"

    # Status
    status = "generated" if valid_mappings else "failed"

    # If skill returned needs_generation (LLM disabled)
    if skill_result.status == MappingAIStatus.NEEDS_GENERATION:
        status = "failed"
        quality_level = "failed"
        quality_messages.insert(0, "LLM chưa sẵn sàng hoặc chưa được cấu hình.")

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    logger.info(
        "mapping_ai.done update_cycle=%s program=%s "
        "mappings=%d retrieved=%d used=%d retrieval=%s quality=%s elapsed_ms=%d",
        update_cycle_id, program_code,
        len(valid_mappings), retrieved_chunk_count, used_chunk_count,
        retrieval_used, quality_level, elapsed_ms,
    )

    return MappingAIResult(
        status=status,
        update_cycle_id=update_cycle_id,
        program_id=program_id,
        program_code=program_code,
        program_name=program_name,
        analysis_mode=analysis_mode,
        mappings=valid_mappings,
        coverage=coverage,
        quality_level=quality_level,
        quality_messages=quality_messages,
        retrieval_used=retrieval_used,
        source_summary=source_summary,
        warnings=all_warnings,
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _build_retrieval_queries(
    *,
    general_objective: str | None,
    objectives: list[dict[str, str]],
    outcomes: list[dict[str, str]],
    program_name: str | None,
) -> list[str]:
    """Build retrieval queries for MATRIX_MAPPING context.

    Up to 3 queries, each ≤ _QUERY_CHAR_LIMIT chars:
      1. Generic structural query (always present).
      2. Compact query from general_objective + a few specific objectives.
      3. Compact query from a few outcomes.
    """
    queries: list[str] = [
        "ma trận liên kết mục tiêu đào tạo chuẩn đầu ra, "
        "cấu trúc chương trình, khối kiến thức, học phần"
    ]

    # Query 2: objective-based
    obj_query = _build_compact_objective_query(general_objective, objectives)
    if obj_query and obj_query not in queries:
        queries.append(obj_query)

    # Query 3: outcome-based
    out_query = _build_compact_outcome_query(outcomes)
    if out_query and out_query not in queries:
        queries.append(out_query)

    return queries[:3]


def _build_compact_objective_query(
    general_objective: str | None,
    objectives: list[dict[str, str]],
) -> str:
    """Build a compact retrieval query from objective content."""
    parts: list[str] = []

    go = (general_objective or "").strip()
    if go:
        # Take first ~300 chars of general objective
        parts.append(go[:300])

    # Add first 3 specific objectives (compact)
    for obj in objectives[:3]:
        text = (obj.get("text") or "").strip()
        if text:
            parts.append(f"{obj.get('code', '')}: {text[:150]}")

    if not parts:
        return ""

    query = " ".join(parts)
    return query[:_QUERY_CHAR_LIMIT].strip()


def _build_compact_outcome_query(
    outcomes: list[dict[str, str]],
) -> str:
    """Build a compact retrieval query from outcome content."""
    parts: list[str] = []

    # Add first 4 outcomes (compact)
    for out in outcomes[:4]:
        text = (out.get("text") or "").strip()
        if text:
            parts.append(f"{out.get('code', '')}: {text[:150]}")

    if not parts:
        return ""

    query = "chuẩn đầu ra " + " ".join(parts)
    return query[:_QUERY_CHAR_LIMIT].strip()


def _select_context_chunks_for_prompt(
    chunks: list[dict[str, Any]],
    *,
    max_total_chars: int,
    per_chunk_limit: int,
) -> list[dict[str, Any]]:
    """Select and truncate context chunks for the LLM prompt budget.

    Input: deduped + score-sorted chunks from retrieval.
    Output: list of chunks with text truncated, within total char budget.
    Preserves relevance order (caller already sorted by score desc).
    """
    selected: list[dict[str, Any]] = []
    total_chars = 0

    for chunk in chunks:
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            break

        raw_text = (chunk.get("text") or "").strip()
        if not raw_text:
            continue

        truncated_text = raw_text[:min(per_chunk_limit, remaining)]
        total_chars += len(truncated_text)

        selected.append({
            **chunk,
            "text": truncated_text,
        })

    return selected


def _dedupe_context_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Deduplicate context chunks by (ai_document_id, chunk_id)."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for chunk in chunks:
        key = f"{chunk.get('ai_document_id')}:{chunk.get('chunk_id')}"
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    # Sort by score descending
    result.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return result
