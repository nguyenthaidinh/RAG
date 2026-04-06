"""
System context orchestrator (Phase 1.1 — Hardened).

Top-level service that ties together:
  - QuestionClassifier → decide what kind of question
  - SystemContextBuilder → fetch relevant context
  - Routing decision → which data planes to use

Phase 1.1 hardening:
  - needs_context is derived from recommended_flags.needs_context_build,
    not from routing booleans. This fixes the inconsistency where
    KNOWLEDGE flags requested user but context was never built.
  - MIXED category respects access signals from classifier.
  - Routing and flags are consistent by construction.

Phase 2 will integrate this into the assistant pipeline via a
feature flag.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.schemas.system_context import SystemContextBundle
from app.services.orchestration.question_classifier import (
    ClassificationResult,
    QuestionCategory,
    QuestionClassifier,
)
from app.services.system_context.context_builder import (
    ContextBuildFlags,
    SystemContextBuilder,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestrationResult:
    """
    Output of the orchestrator's evaluate() method.

    Contains the routing decision + optional context bundle.
    """

    category: QuestionCategory
    classification: ClassificationResult
    should_use_knowledge: bool
    should_use_system_context: bool
    should_use_access_context: bool
    recommended_flags: ContextBuildFlags
    context_bundle: SystemContextBundle | None = None
    notes: tuple[str, ...] = ()
    evaluated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def telemetry_dict(self) -> dict:
        """Safe telemetry — no raw text."""
        return {
            "category": self.category.value,
            "should_use_knowledge": self.should_use_knowledge,
            "should_use_system_context": self.should_use_system_context,
            "should_use_access_context": self.should_use_access_context,
            "has_context_bundle": self.context_bundle is not None,
            "classification": self.classification.telemetry_dict(),
        }


def _build_routing_and_flags(
    category: QuestionCategory,
    classification: ClassificationResult,
) -> tuple[dict[str, bool], ContextBuildFlags]:
    """
    Build routing booleans AND context flags from classification.

    Returns (routing_dict, flags) — they are guaranteed consistent.
    """
    if category == QuestionCategory.KNOWLEDGE:
        return (
            {
                "should_use_knowledge": True,
                "should_use_system_context": False,
                "should_use_access_context": False,
            },
            ContextBuildFlags(
                include_user=False,
                include_tenant=False,
            ),
        )

    if category == QuestionCategory.SYSTEM:
        return (
            {
                "should_use_knowledge": False,
                "should_use_system_context": True,
                "should_use_access_context": False,
            },
            ContextBuildFlags(
                include_user=True,
                include_tenant=True,
                include_stats=True,
                include_records=True,
                include_workflows=True,
            ),
        )

    if category == QuestionCategory.ACCESS:
        return (
            {
                "should_use_knowledge": False,
                "should_use_system_context": False,
                "should_use_access_context": True,
            },
            ContextBuildFlags(
                include_user=True,
                include_tenant=True,
                include_permissions=True,
            ),
        )

    if category == QuestionCategory.MIXED:
        # Phase 1.1: if access signals are present in the mix,
        # include access context — but only if the classifier found
        # access-category signals.
        has_access_signals = any(
            s.startswith("access:") for s in classification.matched_signals
        )
        return (
            {
                "should_use_knowledge": True,
                "should_use_system_context": True,
                "should_use_access_context": has_access_signals,
            },
            ContextBuildFlags(
                include_user=True,
                include_tenant=True,
                include_stats=True,
                include_records=True,
                include_permissions=has_access_signals,
            ),
        )

    # UNKNOWN: knowledge fallback, no context needed
    return (
        {
            "should_use_knowledge": True,
            "should_use_system_context": False,
            "should_use_access_context": False,
        },
        ContextBuildFlags(
            include_user=False,
            include_tenant=False,
        ),
    )


class SystemContextOrchestrator:
    """
    Orchestrates question classification + context building.

    Stateless — all state comes from constructor args.
    """

    __slots__ = ("_classifier", "_context_builder")

    def __init__(
        self,
        *,
        classifier: QuestionClassifier | None = None,
        context_builder: SystemContextBuilder | None = None,
    ) -> None:
        self._classifier = classifier or QuestionClassifier()
        self._context_builder = context_builder

    async def evaluate(
        self,
        *,
        question: str,
        tenant_id: str,
        actor_user_id: str | int,
        build_context: bool = True,
    ) -> OrchestrationResult:
        """
        Evaluate a question and optionally build system context.

        Args:
            question: Raw user question.
            tenant_id: From authenticated user.
            actor_user_id: From authenticated user.
            build_context: If True and question needs system context,
                           actually call the connector.  If False, only
                           classify + route.

        Returns:
            OrchestrationResult with routing decision and optional bundle.
        """
        # ── 1. Classify ──────────────────────────────────────────────
        classification = self._classifier.classify(question)
        category = classification.category

        # ── 2. Build routing + flags (guaranteed consistent) ──────────
        routing, flags = _build_routing_and_flags(category, classification)

        notes: list[str] = []

        # ── 3. Optionally build context bundle ───────────────────────
        # Phase 1.1: needs_context is derived from flags, not routing bools.
        # This ensures KNOWLEDGE (no flags) never triggers a build,
        # and SYSTEM/ACCESS/MIXED always do when flags require it.
        bundle: SystemContextBundle | None = None
        needs_context = flags.needs_context_build

        if build_context and needs_context and self._context_builder is not None:
            try:
                bundle = await self._context_builder.build(
                    tenant_id=tenant_id,
                    actor_user_id=actor_user_id,
                    flags=flags,
                )
                notes.append(f"context_built provider={self._context_builder.provider_name}")
            except Exception:
                logger.warning(
                    "orchestrator.context_build_failed tenant_id=%s category=%s",
                    tenant_id, category.value, exc_info=True,
                )
                notes.append("context_build_failed")
        elif not build_context and needs_context:
            notes.append("context_skipped build_context=false")
        elif needs_context and self._context_builder is None:
            notes.append("no_context_builder_configured")

        logger.info(
            "orchestrator.evaluated tenant_id=%s category=%s "
            "knowledge=%s system_ctx=%s access_ctx=%s "
            "has_bundle=%s confidence=%.2f",
            tenant_id,
            category.value,
            routing.get("should_use_knowledge"),
            routing.get("should_use_system_context"),
            routing.get("should_use_access_context"),
            bundle is not None,
            classification.confidence,
        )

        return OrchestrationResult(
            category=category,
            classification=classification,
            should_use_knowledge=routing.get("should_use_knowledge", True),
            should_use_system_context=routing.get("should_use_system_context", False),
            should_use_access_context=routing.get("should_use_access_context", False),
            recommended_flags=flags,
            context_bundle=bundle,
            notes=tuple(notes),
        )
