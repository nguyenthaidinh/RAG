"""Tests for R6.5 Objective Quality Pack."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services.ctdt_objective_context_service import (
    ContextItem,
    ContextPackSourceSummary,
    ObjectiveUpdateContextPack,
    RoleCoverageItem,
)
from app.services.ctdt_objective_quality_service import (
    ObjectiveAdaptedResult,
    adapt_objective_payload,
    build_debug_context,
    check_objective_quality,
    deduplicate_contexts,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_ctx_item(
    *,
    ai_document_id: int = 1,
    document_role: str = "current_curriculum",
    filename: str = "ctdt.pdf",
    text: str = "Mục tiêu đào tạo kỹ sư CNTT",
    chunk_index: int = 0,
    score: float = 0.85,
) -> ContextItem:
    return ContextItem(
        ai_document_id=ai_document_id,
        external_file_id=f"ext-{ai_document_id}",
        filename=filename,
        document_role=document_role,
        chunk_id=ai_document_id * 100000 + chunk_index,
        chunk_index=chunk_index,
        score=score,
        text=text,
        source={"update_cycle_id": "15", "program_code": "7480201"},
    )


def _make_context_pack(
    *,
    with_contexts: bool = True,
    missing_info: list | None = None,
) -> ObjectiveUpdateContextPack:
    pack = ObjectiveUpdateContextPack(
        update_cycle_id="15",
        program_code="7480201",
        program_name="Công nghệ thông tin",
    )
    if with_contexts:
        pack.current_objective_contexts = [
            _make_ctx_item(ai_document_id=1, document_role="current_curriculum"),
        ]
        pack.direction_contexts = [
            _make_ctx_item(ai_document_id=2, document_role="direction_decision",
                           filename="quyet_dinh.pdf",
                           text="Yêu cầu cập nhật CTĐT theo định hướng mới"),
        ]
        pack.evidence_contexts = [
            _make_ctx_item(ai_document_id=3, document_role="survey_evidence",
                           filename="khao_sat.pdf",
                           text="Kết quả khảo sát doanh nghiệp"),
        ]
        for key, roles, count in [
            ("current_objective", ["current_curriculum"], 1),
            ("direction", ["direction_decision"], 1),
            ("legal", ["legal_regulation"], 0),
            ("evidence", ["survey_evidence", "meeting_report"], 1),
            ("comparison", ["comparison_report"], 0),
        ]:
            pack.role_coverage[key] = RoleCoverageItem(
                document_roles=roles, context_count=count,
                documents_used=[], status="available" if count > 0 else "missing",
                scoped_document_count=count, retrieval_status="ok",
            )
    else:
        for key, roles in [
            ("current_objective", ["current_curriculum"]),
            ("direction", ["direction_decision"]),
            ("legal", ["legal_regulation"]),
            ("evidence", ["survey_evidence", "meeting_report"]),
            ("comparison", ["comparison_report"]),
        ]:
            pack.role_coverage[key] = RoleCoverageItem(
                document_roles=roles, context_count=0, documents_used=[],
                status="missing", scoped_document_count=0, retrieval_status="ok",
            )
    pack.missing_information = missing_info or []
    pack.source_summary = ContextPackSourceSummary(
        total_contexts=3 if with_contexts else 0,
        documents_used=[1, 2, 3] if with_contexts else [],
        role_groups_retrieved=["current_objective", "direction", "evidence"],
        latency_ms=50,
    )
    return pack


# ══════════════════════════════════════════════════════════════════════
# 1. Adapter tests
# ══════════════════════════════════════════════════════════════════════


class TestAdaptObjectivePayload:
    def test_direct_fields_preferred(self):
        payload = {
            "general_objective_text": "Đào tạo kỹ sư CNTT có năng lực.",
            "specific_objective_texts": ["Mục tiêu 1", "Mục tiêu 2", "Mục tiêu 3", "Mục tiêu 4"],
            "proposed_objectives": [{"objective_type": "general_objective", "proposed_content": "Old"}],
        }
        result = adapt_objective_payload(payload=payload, program_name="CNTT")
        assert result.general_objective == "Đào tạo kỹ sư CNTT có năng lực."
        assert len(result.specific_objectives) == 4

    def test_classify_from_proposed(self):
        payload = {
            "proposed_objectives": [
                {"objective_type": "general_objective", "proposed_content": "Mục tiêu chung CNTT"},
                {"objective_type": "specific_objective", "proposed_content": "Cụ thể 1"},
                {"objective_type": "specific_objective", "proposed_content": "Cụ thể 2"},
            ],
        }
        result = adapt_objective_payload(payload=payload)
        assert "CNTT" in result.general_objective
        assert len(result.specific_objectives) == 2

    def test_heuristic_when_no_general(self):
        payload = {
            "proposed_objectives": [
                {"objective_type": "specific_objective", "proposed_content": "Dài nhất " * 20},
                {"objective_type": "specific_objective", "proposed_content": "Ngắn"},
            ],
        }
        result = adapt_objective_payload(payload=payload)
        assert result.general_objective  # longest becomes general
        assert len(result.specific_objectives) == 1
        assert any("tự phân loại" in w for w in result.warnings)

    def test_empty_payload(self):
        result = adapt_objective_payload(payload={}, generation_status="generated")
        assert result.general_objective == ""
        assert result.specific_objectives == []
        assert any("không sinh được" in w.lower() for w in result.warnings)

    def test_raw_payload_preserved(self):
        payload = {"proposed_objectives": [], "some_extra": "data"}
        result = adapt_objective_payload(payload=payload)
        assert result.raw_payload == payload
        assert "some_extra" in result.raw_payload


# ══════════════════════════════════════════════════════════════════════
# 2. Quality check tests
# ══════════════════════════════════════════════════════════════════════


class TestCheckObjectiveQuality:
    def test_empty_general_warns(self):
        warnings = check_objective_quality(
            general_objective="", specific_objectives=["a", "b", "c", "d"],
        )
        assert any("chưa được sinh" in w for w in warnings)

    def test_few_specifics_warns(self):
        warnings = check_objective_quality(
            general_objective="Đào tạo kỹ sư CNTT", specific_objectives=["a", "b"],
        )
        assert any("mục tiêu cụ thể" in w for w in warnings)

    def test_too_many_specifics_warns(self):
        warnings = check_objective_quality(
            general_objective="Đào tạo kỹ sư CNTT",
            specific_objectives=[f"Mục tiêu {i}" for i in range(8)],
        )
        assert any("mục tiêu cụ thể" in w for w in warnings)

    def test_generic_content_warns(self):
        warnings = check_objective_quality(
            general_objective="Có kiến thức và kỹ năng cần thiết, đáp ứng nhu cầu xã hội.",
            specific_objectives=["a", "b", "c", "d"],
        )
        assert any("chung chung" in w for w in warnings)

    def test_no_evidence_warns(self):
        warnings = check_objective_quality(
            general_objective="Đào tạo kỹ sư CNTT",
            specific_objectives=["a", "b", "c", "d"],
            has_evidence_context=False,
        )
        assert any("khảo sát" in w for w in warnings)

    def test_no_curriculum_warns(self):
        warnings = check_objective_quality(
            general_objective="Đào tạo kỹ sư CNTT",
            specific_objectives=["a", "b", "c", "d"],
            has_current_curriculum_context=False,
        )
        assert any("hiện hành" in w for w in warnings)

    def test_good_output_no_critical_warnings(self):
        warnings = check_objective_quality(
            general_objective="Đào tạo kỹ sư công nghệ thông tin có năng lực chuyên môn sâu.",
            specific_objectives=[
                "Kiến thức nền tảng CNTT", "Kỹ năng lập trình",
                "Năng lực giải quyết vấn đề", "Đạo đức nghề nghiệp",
            ],
            program_name="Công nghệ thông tin",
        )
        # Should not have critical warnings
        assert not any("chưa được sinh" in w for w in warnings)
        assert not any("khuyến nghị 4-6" in w for w in warnings)

    def test_outcome_like_specifics_warns(self):
        warnings = check_objective_quality(
            general_objective="Đào tạo kỹ sư CNTT",
            specific_objectives=[
                "Phân tích, thiết kế, triển khai, đánh giá hệ thống phần mềm",
                "Vận dụng, xây dựng, kiểm thử, triển khai ứng dụng web",
                "Mục tiêu bình thường",
                "Đạo đức nghề nghiệp",
            ],
        )
        assert any("chuẩn đầu ra" in w for w in warnings)


# ══════════════════════════════════════════════════════════════════════
# 3. Debug context tests
# ══════════════════════════════════════════════════════════════════════


class TestBuildDebugContext:
    def test_debug_has_required_fields(self):
        pack = _make_context_pack()
        debug = build_debug_context(
            context_pack=pack,
            queries_used=["mục tiêu đào tạo", "PEO"],
            fallback_used=False,
        )
        assert "queries" in debug
        assert "used_chunks" in debug
        assert "missing_roles" in debug
        assert "fallback_used" in debug
        assert "context_char_count" in debug

    def test_debug_text_preview_limited(self):
        pack = _make_context_pack()
        pack.current_objective_contexts[0].text = "A" * 1000
        debug = build_debug_context(context_pack=pack)
        for chunk in debug["used_chunks"]:
            assert len(chunk["text_preview"]) <= 401  # 400 + "…"

    def test_debug_false_returns_none_in_service(self):
        """When debug_context=False, service should return debug=None."""
        # This is tested via the orchestrator integration
        pass

    def test_missing_roles_detected(self):
        pack = _make_context_pack()
        debug = build_debug_context(context_pack=pack)
        assert "legal" in debug["missing_roles"]
        assert "comparison" in debug["missing_roles"]

    def test_dedup_chunks(self):
        pack = _make_context_pack()
        # Add duplicate
        pack.direction_contexts.append(
            _make_ctx_item(ai_document_id=1, chunk_index=0, score=0.5),
        )
        debug = build_debug_context(context_pack=pack)
        ids = [(c["document_id"], c["chunk_index"]) for c in debug["used_chunks"]]
        assert len(ids) == len(set(ids))


# ══════════════════════════════════════════════════════════════════════
# 4. Latest draft endpoint tests
# ══════════════════════════════════════════════════════════════════════


class TestLatestObjectiveDraftEndpoint:
    @pytest.mark.asyncio
    async def test_draft_found_returns_200(self):
        from app.api.v1.ctdt import get_latest_objective_draft

        mock_draft = MagicMock()
        mock_draft.id = 42
        mock_draft.status = "draft"
        mock_draft.program_name = "CNTT"
        mock_draft.program_code = "7480201"
        mock_draft.result_payload = {
            "proposed_objectives": [
                {"objective_type": "general_objective",
                 "proposed_content": "Đào tạo kỹ sư CNTT"},
                {"objective_type": "specific_objective",
                 "proposed_content": "Kiến thức nền tảng"},
                {"objective_type": "specific_objective",
                 "proposed_content": "Kỹ năng thực hành"},
                {"objective_type": "specific_objective",
                 "proposed_content": "Đạo đức nghề nghiệp"},
                {"objective_type": "specific_objective",
                 "proposed_content": "Tự học nghiên cứu"},
            ],
        }
        mock_draft.created_at = "2026-05-20"
        mock_draft.updated_at = "2026-05-20"

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            return_value=mock_draft,
        ):
            response = await get_latest_objective_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        assert response.success is True
        assert response.data["draft_id"] == 42
        assert response.data["general_objective"]
        assert isinstance(response.data["specific_objectives"], list)

    @pytest.mark.asyncio
    async def test_no_draft_returns_404(self):
        from app.api.v1.ctdt import get_latest_objective_draft

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            return_value=None,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_latest_objective_draft(
                    update_cycle_id="15",
                    request=AsyncMock(),
                    db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail["code"] == "objective_draft_not_found"

    @pytest.mark.asyncio
    async def test_invalid_mode_returns_422(self):
        from app.api.v1.ctdt import get_latest_objective_draft

        with pytest.raises(HTTPException) as exc_info:
            await get_latest_objective_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                analysis_mode="invalid",
            )

        assert exc_info.value.status_code == 422
        assert exc_info.value.detail["code"] == "invalid_analysis_mode"

    @pytest.mark.asyncio
    async def test_db_error_returns_500(self):
        from app.api.v1.ctdt import get_latest_objective_draft

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            side_effect=RuntimeError("DB down"),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_latest_objective_draft(
                    update_cycle_id="15",
                    request=AsyncMock(),
                    db=AsyncMock(),
                    user=SimpleNamespace(tenant_id="t1", id=7),
                )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail["code"] == "objective_draft_latest_error"


# ══════════════════════════════════════════════════════════════════════
# 5. Fallback tests
# ══════════════════════════════════════════════════════════════════════


class TestFallbackRole:
    @pytest.mark.asyncio
    async def test_no_fallback_when_contexts_exist(self):
        """current_curriculum has context → no fallback triggered."""
        from app.services.ctdt_objective_context_service import (
            build_objective_update_context_pack,
        )
        from app.services.ctdt_retrieval_service import CTDTRetrievalResult

        mock_result = CTDTRetrievalResult(
            query="test", update_cycle_id="15", program_code="7480201",
            task_type="objective_suggestion", document_roles_used=["current_curriculum"],
            contexts=[MagicMock(
                ai_document_id=1, external_file_id="ext-1", filename="ctdt.pdf",
                document_role="current_curriculum", chunk_id=100001, chunk_index=1,
                score=0.9, text="Mục tiêu đào tạo CNTT", source={},
            )],
            scoped_document_count=1, latency_ms=10,
        )

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            return_value=mock_result,
        ):
            pack = await build_objective_update_context_pack(
                AsyncMock(), tenant_id="t1", user_id=7,
                update_cycle_id="15", program_code="7480201",
            )

        assert pack.fallback_used is False

    @pytest.mark.asyncio
    async def test_fallback_stays_within_update_cycle(self):
        """Fallback must not cross update_cycle boundary."""
        from app.services.ctdt_objective_context_service import (
            _fallback_retrieve,
        )
        from app.services.ctdt_retrieval_service import CTDTRetrievalResult

        calls = []

        async def mock_retrieve(db, **kwargs):
            calls.append(kwargs)
            return CTDTRetrievalResult(
                query=kwargs["query"], update_cycle_id=kwargs["update_cycle_id"],
                program_code=kwargs.get("program_code"),
                task_type="objective_suggestion",
                document_roles_used=kwargs.get("document_roles") or [],
                contexts=[], scoped_document_count=0, latency_ms=5,
            )

        with patch(
            "app.services.ctdt_objective_context_service.ctdt_retrieve",
            side_effect=mock_retrieve,
        ):
            await _fallback_retrieve(
                AsyncMock(), tenant_id="t1", user_id=7,
                queries=["mục tiêu đào tạo"],
                update_cycle_id="15", program_code="7480201",
                program_id=None, top_k=5,
            )

        # All calls must have the same update_cycle_id
        for call in calls:
            assert call["update_cycle_id"] == "15"


# ══════════════════════════════════════════════════════════════════════
# 6. Context char limits (R6.5 YC7)
# ══════════════════════════════════════════════════════════════════════


class TestContextCharLimits:
    def test_variable_limits_applied(self):
        from app.services.ctdt_skills.objective_update_skill import (
            _build_user_prompt, _CONTEXT_CHAR_LIMITS,
        )
        pack = _make_context_pack()
        pack.current_objective_contexts[0].text = "A" * 5000

        prompt, _ = _build_user_prompt(
            program_name="CNTT", program_code="7480201",
            update_cycle_id="15", context_pack=pack,
        )

        # current_objective limit is 2500, text was 5000 → should be truncated
        limit = _CONTEXT_CHAR_LIMITS["current_objective"]
        # The prompt should contain at most `limit` consecutive A's
        longest_a_run = max(
            (len(s) for s in prompt.split("\n") if set(s) == {"A"}),
            default=0,
        )
        assert longest_a_run <= limit


# ══════════════════════════════════════════════════════════════════════
# 7. Deduplication
# ══════════════════════════════════════════════════════════════════════


class TestDeduplication:
    def test_dedup_keeps_highest_score(self):
        items = [
            _make_ctx_item(ai_document_id=1, chunk_index=0, score=0.7),
            _make_ctx_item(ai_document_id=1, chunk_index=0, score=0.9),
            _make_ctx_item(ai_document_id=2, chunk_index=0, score=0.8),
        ]
        result = deduplicate_contexts(items)
        assert len(result) == 2
        scores = {(c.ai_document_id, c.score) for c in result}
        assert (1, 0.9) in scores


# ══════════════════════════════════════════════════════════════════════
# 8. Backward compatibility
# ══════════════════════════════════════════════════════════════════════


class TestBackwardCompat:
    @pytest.mark.asyncio
    async def test_post_endpoint_still_has_payload_field(self):
        from app.api.v1.ctdt import ObjectiveDraftRequest, generate_objectives_draft
        from app.services.ctdt_objective_update_service import (
            ObjectiveDraftResult, ObjectiveSourceSummary,
        )

        mock_result = ObjectiveDraftResult(
            update_cycle_id="15", program_code="7480201", program_name="CNTT",
            draft_type="objective_update", draft_id=None, draft_saved=False,
            payload={"proposed_objectives": []},
            context_pack_summary={"role_coverage": {}, "missing_information": []},
            source_summary=ObjectiveSourceSummary(
                contexts_count=0, documents_used=[], tasks_executed=["objective_update"],
                latency_ms=50,
            ),
            generation_status="generated", warnings=[],
        )

        with patch(
            "app.services.ctdt_objective_update_service.generate_objective_update_draft",
            return_value=mock_result,
        ):
            response = await generate_objectives_draft(
                body=ObjectiveDraftRequest(update_cycle_id="15"),
                request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        # Old fields still present
        assert hasattr(response, "payload")
        assert hasattr(response, "context_pack_summary")
        assert hasattr(response, "generation_status")
        # New fields also present
        assert hasattr(response, "general_objective")
        assert hasattr(response, "specific_objectives")


# ══════════════════════════════════════════════════════════════════════
# 9. Skill R6.5 fields
# ══════════════════════════════════════════════════════════════════════


class TestSkillR65Fields:
    @pytest.mark.asyncio
    async def test_payload_has_new_fields(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        llm_data = {
            "general_objective_text": "Đào tạo kỹ sư CNTT chất lượng cao.",
            "specific_objective_texts": ["KT nền tảng", "KN thực hành", "NL ứng dụng", "ĐĐ nghề nghiệp"],
            "objective_update_strategy": {"summary": "Test", "main_drivers": [], "human_review_required": True},
            "current_objective_analysis": [],
            "proposed_objectives": [],
            "alignment_notes": [],
            "objective_quality_review": {},
            "missing_information": [],
            "risks": [],
            "next_actions": [],
        }

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(update_cycle_id="15", context_pack=pack)

        assert result.payload.general_objective_text == "Đào tạo kỹ sư CNTT chất lượng cao."
        assert len(result.payload.specific_objective_texts) == 4


# ══════════════════════════════════════════════════════════════════════
# 10. R6.5-PATCH-1: Service stores _flat when save_draft=True
# ══════════════════════════════════════════════════════════════════════


class TestServiceSavesDraftFlat:
    @pytest.mark.asyncio
    async def test_save_draft_stores_flat_in_result_payload(self):
        """When save_draft=True, result_payload must contain _flat block."""
        from app.services.ctdt_objective_update_service import (
            generate_objective_update_draft,
        )
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateResult,
            ObjectiveUpdatePayload,
            ObjectiveUpdateStatus,
        )

        # Track what gets saved to DB
        saved_payloads = []

        class FakeDraft:
            id = 99
            status = "draft"

        fake_draft = FakeDraft()

        async def fake_flush():
            pass

        async def fake_refresh(obj):
            obj.id = fake_draft.id

        async def fake_commit():
            pass

        db = AsyncMock()
        db.flush = fake_flush
        db.refresh = fake_refresh
        db.commit = fake_commit

        original_add = db.add

        def capture_add(obj):
            saved_payloads.append(obj.result_payload)
            return original_add(obj)

        db.add = capture_add

        mock_skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.GENERATED,
            payload=ObjectiveUpdatePayload(
                proposed_objectives=[
                    {"objective_type": "general_objective",
                     "proposed_content": "Đào tạo kỹ sư CNTT"},
                    {"objective_type": "specific_objective",
                     "proposed_content": "Kiến thức nền tảng"},
                    {"objective_type": "specific_objective",
                     "proposed_content": "Kỹ năng thực hành"},
                    {"objective_type": "specific_objective",
                     "proposed_content": "Đạo đức nghề nghiệp"},
                    {"objective_type": "specific_objective",
                     "proposed_content": "Tự học nghiên cứu"},
                ],
                general_objective_text="Đào tạo kỹ sư CNTT chất lượng cao.",
                specific_objective_texts=["KT", "KN", "NL", "ĐĐ"],
            ),
            warnings=[],
        )

        pack = _make_context_pack()

        with patch(
            "app.services.ctdt_objective_update_service.build_objective_update_context_pack",
            return_value=pack,
        ), patch(
            "app.services.ctdt_skills.objective_update_skill.ObjectiveUpdateSkill.run",
            return_value=mock_skill_result,
        ):
            result = await generate_objective_update_draft(
                db,
                tenant_id="t1",
                user_id=7,
                update_cycle_id="15",
                program_code="7480201",
                program_name="CNTT",
                save_draft=True,
            )

        assert result.draft_saved is True
        assert len(saved_payloads) == 1

        stored = saved_payloads[0]
        assert "_flat" in stored
        flat = stored["_flat"]
        assert isinstance(flat.get("general_objective"), str)
        assert flat["general_objective"]  # non-empty
        assert isinstance(flat.get("specific_objectives"), list)
        assert isinstance(flat.get("warnings"), list)
        assert isinstance(flat.get("source_summary"), dict)
        # debug must NOT be stored in DB
        assert "debug" not in flat
        assert "used_chunks" not in flat


# ══════════════════════════════════════════════════════════════════════
# 11. R6.5-PATCH-1: Latest endpoint prefers _flat
# ══════════════════════════════════════════════════════════════════════


class TestLatestEndpointPrefersFlat:
    @pytest.mark.asyncio
    async def test_uses_flat_when_available(self):
        """If result_payload has _flat, latest endpoint must use it
        instead of running adapter on raw proposed_objectives."""
        from app.api.v1.ctdt import get_latest_objective_draft

        mock_draft = MagicMock()
        mock_draft.id = 77
        mock_draft.status = "draft"
        mock_draft.program_name = "CNTT"
        mock_draft.program_code = "7480201"
        mock_draft.result_payload = {
            "_flat": {
                "general_objective": "Flat general",
                "specific_objectives": ["A", "B", "C", "D"],
                "source_summary": {"used_documents": [1]},
                "warnings": ["Flat warning"],
            },
            "proposed_objectives": [
                {"objective_type": "general_objective",
                 "proposed_content": "Raw general — should NOT appear"},
            ],
        }
        mock_draft.created_at = "2026-05-21"
        mock_draft.updated_at = "2026-05-21"

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            return_value=mock_draft,
        ):
            response = await get_latest_objective_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        assert response.data["general_objective"] == "Flat general"
        assert response.data["specific_objectives"] == ["A", "B", "C", "D"]
        assert "Flat warning" in response.data["warnings"]
        assert response.data["source_summary"] == {"used_documents": [1]}
        # Must NOT contain the raw fallback value
        assert "Raw general" not in response.data["general_objective"]


# ══════════════════════════════════════════════════════════════════════
# 12. R6.5-PATCH-1: Latest endpoint fallback for old draft
# ══════════════════════════════════════════════════════════════════════


class TestLatestEndpointFallbackOldDraft:
    @pytest.mark.asyncio
    async def test_fallback_adapter_when_no_flat(self):
        """Old drafts without _flat should still adapt via adapter."""
        from app.api.v1.ctdt import get_latest_objective_draft

        mock_draft = MagicMock()
        mock_draft.id = 42
        mock_draft.status = "draft"
        mock_draft.program_name = "CNTT"
        mock_draft.program_code = "7480201"
        # Old draft: no _flat key at all
        mock_draft.result_payload = {
            "proposed_objectives": [
                {"objective_type": "general_objective",
                 "proposed_content": "Đào tạo kỹ sư CNTT"},
                {"objective_type": "specific_objective",
                 "proposed_content": "Kiến thức nền tảng"},
                {"objective_type": "specific_objective",
                 "proposed_content": "Kỹ năng thực hành"},
                {"objective_type": "specific_objective",
                 "proposed_content": "Đạo đức nghề nghiệp"},
                {"objective_type": "specific_objective",
                 "proposed_content": "Tự học nghiên cứu"},
            ],
        }
        mock_draft.created_at = "2026-05-20"
        mock_draft.updated_at = "2026-05-20"

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            return_value=mock_draft,
        ):
            response = await get_latest_objective_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        assert response.success is True
        assert response.data["general_objective"]  # non-empty from adapter
        assert isinstance(response.data["specific_objectives"], list)
        assert response.data["debug"] is None


# ══════════════════════════════════════════════════════════════════════
# 13. R6.5-PATCH-1: Latest endpoint has no debug_context param
# ══════════════════════════════════════════════════════════════════════


class TestLatestEndpointNoDebugParam:
    def test_signature_has_no_debug_context(self):
        """get_latest_objective_draft must not accept debug_context."""
        import inspect
        from app.api.v1.ctdt import get_latest_objective_draft

        sig = inspect.signature(get_latest_objective_draft)
        param_names = list(sig.parameters.keys())
        assert "debug_context" not in param_names

    @pytest.mark.asyncio
    async def test_response_always_has_debug_none(self):
        """Response must still include debug=None for client compatibility."""
        from app.api.v1.ctdt import get_latest_objective_draft

        mock_draft = MagicMock()
        mock_draft.id = 1
        mock_draft.status = "draft"
        mock_draft.program_name = "CNTT"
        mock_draft.program_code = "7480201"
        mock_draft.result_payload = {
            "_flat": {
                "general_objective": "Test",
                "specific_objectives": ["A", "B", "C", "D"],
                "source_summary": {},
                "warnings": [],
            },
        }
        mock_draft.created_at = "2026-05-21"
        mock_draft.updated_at = "2026-05-21"

        with patch(
            "app.services.ctdt_analysis_draft_service.get_latest_analysis_draft",
            return_value=mock_draft,
        ):
            response = await get_latest_objective_draft(
                update_cycle_id="15",
                request=AsyncMock(),
                db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
            )

        assert "debug" in response.data
        assert response.data["debug"] is None

