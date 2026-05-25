"""Tests for R6.5C — Dynamic objective_count for Mục tiêu đào tạo."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.ctdt_objective_context_service import (
    ContextItem,
    ContextPackSourceSummary,
    ObjectiveUpdateContextPack,
    RoleCoverageItem,
)
from app.services.ctdt_skills.objective_update_skill import (
    ObjectiveUpdatePayload,
    ObjectiveUpdateResult,
    ObjectiveUpdateStatus,
    compute_objective_allocation,
    _ALLOCATION_TABLE,
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


def _make_context_pack(*, with_contexts: bool = True) -> ObjectiveUpdateContextPack:
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
    pack.missing_information = []
    pack.source_summary = ContextPackSourceSummary(
        total_contexts=3 if with_contexts else 0,
        documents_used=[1, 2, 3] if with_contexts else [],
        role_groups_retrieved=["current_objective", "direction", "evidence"],
        latency_ms=50,
    )
    return pack


def _make_structured_llm_response(objective_count: int = 6) -> str:
    """Build a structured LLM response with M-codes for given count."""
    alloc = compute_objective_allocation(objective_count)
    objectives = []
    for group, codes in alloc.items():
        for code in codes:
            objectives.append({
                "code": code,
                "group": group,
                "text": f"Mục tiêu {code} thuộc nhóm {group}.",
            })
    return json.dumps({
        "general_objective": "Mục tiêu của chương trình là đào tạo kỹ sư CNTT.",
        "specific_objectives": objectives,
        "evidence_quality": "moderate",
        "warnings": [],
    })


# ══════════════════════════════════════════════════════════════════════
# 1. Allocation table tests
# ══════════════════════════════════════════════════════════════════════


def _make_structured_objectives(objective_count: int = 6) -> list[dict]:
    alloc = compute_objective_allocation(objective_count)
    objectives = []
    for group, codes in alloc.items():
        for code in codes:
            objectives.append({
                "code": code,
                "group": group,
                "text": f"Mục tiêu {code} thuộc nhóm {group}.",
            })
    return objectives


def _make_structured_llm_response_from_objectives(
    objectives: list[dict],
    *,
    evidence_quality: str = "moderate",
) -> str:
    return json.dumps({
        "general_objective": (
            "Mục tiêu của chương trình là đào tạo kỹ sư Công nghệ thông tin."
        ),
        "specific_objectives": objectives,
        "evidence_quality": evidence_quality,
        "warnings": [],
    })


async def _run_skill_with_objectives(
    objectives: list[dict],
    *,
    objective_count: int = 8,
    evidence_quality: str = "moderate",
):
    from app.services.ctdt_skills.objective_update_skill import (
        ObjectiveUpdateSkill,
    )

    skill = ObjectiveUpdateSkill()
    pack = _make_context_pack()

    with patch.object(
        skill,
        "_call_openai",
        return_value=_make_structured_llm_response_from_objectives(
            objectives,
            evidence_quality=evidence_quality,
        ),
    ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
        ms.SYNTHESIS_ENABLED = True
        ms.OPENAI_API_KEY = "sk-test"
        return await skill.run(
            update_cycle_id="15",
            context_pack=pack,
            objective_count=objective_count,
        )


def _make_objective_payload_for_service(
    *,
    objective_count: int = 8,
    objectives: list[dict] | None = None,
    evidence_quality: str = "moderate",
) -> ObjectiveUpdatePayload:
    return ObjectiveUpdatePayload(
        general_objective="Mục tiêu của chương trình là đào tạo kỹ sư Công nghệ thông tin.",
        general_objective_text="Mục tiêu của chương trình là đào tạo kỹ sư Công nghệ thông tin.",
        specific_objectives_structured=objectives or _make_structured_objectives(objective_count),
        objective_count=objective_count,
        format_profile="tay_nguyen_ctdt_dynamic_m_objectives",
        evidence_quality=evidence_quality,
    )


async def _run_service_with_skill_result(
    skill_result: ObjectiveUpdateResult,
    *,
    objective_count: int = 8,
    save_draft: bool = False,
    db=None,
):
    from app.services.ctdt_objective_update_service import (
        generate_objective_update_draft,
    )

    pack = _make_context_pack()
    db = db or AsyncMock()

    with patch(
        "app.services.ctdt_objective_update_service.build_objective_update_context_pack",
        return_value=pack,
    ), patch(
        "app.services.ctdt_skills.objective_update_skill.ObjectiveUpdateSkill.run",
        return_value=skill_result,
    ):
        return await generate_objective_update_draft(
            db,
            tenant_id="t1",
            user_id=7,
            update_cycle_id="15",
            program_code="7480201",
            program_name="Công nghệ thông tin",
            objective_count=objective_count,
            save_draft=save_draft,
        )


class TestAllocationTable:
    def test_all_counts_4_to_8(self):
        for count in range(4, 9):
            alloc = compute_objective_allocation(count)
            all_codes = []
            for codes in alloc.values():
                all_codes.extend(codes)
            # Must have exactly `count` codes
            assert len(all_codes) == count, f"count={count} got {len(all_codes)}"
            # Must start at M1 and end at M{count}
            assert all_codes[0] == "M1"
            assert all_codes[-1] == f"M{count}"
            # No gaps, no duplicates
            expected = [f"M{i}" for i in range(1, count + 1)]
            assert all_codes == expected
            # Last group must be foreign_language_it
            assert "foreign_language_it" in alloc
            assert alloc["foreign_language_it"][-1] == f"M{count}"

    def test_clamp_below_4(self):
        alloc = compute_objective_allocation(2)
        all_codes = []
        for codes in alloc.values():
            all_codes.extend(codes)
        assert len(all_codes) == 4  # clamped to 4

    def test_clamp_above_8(self):
        alloc = compute_objective_allocation(12)
        all_codes = []
        for codes in alloc.values():
            all_codes.extend(codes)
        assert len(all_codes) == 8  # clamped to 8


# ══════════════════════════════════════════════════════════════════════
# 2. Skill with objective_count
# ══════════════════════════════════════════════════════════════════════


class TestSkillObjectiveCount:
    @pytest.mark.asyncio
    async def test_objective_count_5_produces_m1_to_m5(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai",
            return_value=_make_structured_llm_response(5),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
                objective_count=5,
            )

        assert result.status == "generated"
        structured = result.payload.specific_objectives_structured
        assert len(structured) == 5
        assert structured[0]["code"] == "M1"
        assert structured[-1]["code"] == "M5"
        assert structured[-1]["group"] == "foreign_language_it"
        # No M6
        codes = [o["code"] for o in structured]
        assert "M6" not in codes

    @pytest.mark.asyncio
    async def test_objective_count_8_produces_m1_to_m8(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai",
            return_value=_make_structured_llm_response(8),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
                objective_count=8,
            )

        assert result.status == "generated"
        structured = result.payload.specific_objectives_structured
        assert len(structured) == 8
        assert structured[-1]["code"] == "M8"
        assert structured[-1]["group"] == "foreign_language_it"

    @pytest.mark.asyncio
    async def test_default_objective_count_is_6(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai",
            return_value=_make_structured_llm_response(6),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            # No objective_count param → default 6
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        assert result.payload.objective_count == 6
        assert len(result.payload.specific_objectives_structured) == 6


# ══════════════════════════════════════════════════════════════════════
# 3. Structured specific_objectives format
# ══════════════════════════════════════════════════════════════════════


class TestStrictObjectiveContract:
    @pytest.mark.asyncio
    async def test_objective_count_4_exact_contract_generated(self):
        result = await _run_skill_with_objectives(
            _make_structured_objectives(4),
            objective_count=4,
        )

        assert result.status == ObjectiveUpdateStatus.GENERATED
        assert len(result.payload.specific_objectives_structured) == 4

    @pytest.mark.asyncio
    async def test_objective_count_8_exact_contract_generated(self):
        result = await _run_skill_with_objectives(
            _make_structured_objectives(8),
            objective_count=8,
        )

        assert result.status == ObjectiveUpdateStatus.GENERATED
        assert len(result.payload.specific_objectives_structured) == 8

    @pytest.mark.asyncio
    async def test_objective_count_8_ai_returns_m1_to_m5_failed(self):
        result = await _run_skill_with_objectives(
            _make_structured_objectives(5),
            objective_count=8,
        )

        assert result.status == ObjectiveUpdateStatus.FAILED
        assert any("5/8" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_objective_count_8_ai_returns_m1_to_m9_failed(self):
        objectives = _make_structured_objectives(8)
        objectives.append({
            "code": "M9",
            "group": "foreign_language_it",
            "text": "Mục tiêu M9 sinh dư.",
        })

        result = await _run_skill_with_objectives(objectives, objective_count=8)

        assert result.status == ObjectiveUpdateStatus.FAILED
        assert any("9/8" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_objective_count_8_missing_m4_failed(self):
        objectives = [
            obj for obj in _make_structured_objectives(8)
            if obj["code"] != "M4"
        ]

        result = await _run_skill_with_objectives(objectives, objective_count=8)

        assert result.status == ObjectiveUpdateStatus.FAILED
        assert any("M1 đến M8" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_objective_count_8_duplicate_m3_failed(self):
        objectives = _make_structured_objectives(8)
        objectives[3]["code"] = "M3"
        objectives[3]["group"] = "knowledge"

        result = await _run_skill_with_objectives(objectives, objective_count=8)

        assert result.status == ObjectiveUpdateStatus.FAILED
        assert any("trùng" in warning and "M3" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_objective_count_8_wrong_group_failed(self):
        objectives = _make_structured_objectives(8)
        objectives[4]["group"] = "knowledge"

        result = await _run_skill_with_objectives(objectives, objective_count=8)

        assert result.status == ObjectiveUpdateStatus.FAILED
        assert any(
            "M5" in warning and "skills_attitude" in warning
            for warning in result.warnings
        )

    @pytest.mark.asyncio
    async def test_empty_objective_text_failed(self):
        objectives = _make_structured_objectives(8)
        objectives[2]["text"] = " "

        result = await _run_skill_with_objectives(objectives, objective_count=8)

        assert result.status == ObjectiveUpdateStatus.FAILED
        assert any("M3" in warning and "rỗng" in warning for warning in result.warnings)

    @pytest.mark.asyncio
    async def test_missing_structured_specific_objectives_failed(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()
        llm_data = {
            "general_objective_text": "Mục tiêu chung.",
            "specific_objective_texts": ["Mục tiêu 1", "Mục tiêu 2"],
            "evidence_quality": "moderate",
            "warnings": [],
        }

        with patch.object(
            skill,
            "_call_openai",
            return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15",
                context_pack=pack,
                objective_count=8,
            )

        assert result.status == ObjectiveUpdateStatus.FAILED
        assert any(
            "cấu trúc mục tiêu bắt buộc M1 đến M8" in warning
            for warning in result.warnings
        )


class TestStructuredObjectives:
    @pytest.mark.asyncio
    async def test_each_has_code_group_text(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai",
            return_value=_make_structured_llm_response(6),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
                objective_count=6,
            )

        for obj in result.payload.specific_objectives_structured:
            assert "code" in obj
            assert "group" in obj
            assert "text" in obj
            assert obj["group"] in ("knowledge", "skills_attitude", "foreign_language_it")

    @pytest.mark.asyncio
    async def test_last_group_is_foreign_language_it(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        for count in [4, 5, 6, 7, 8]:
            with patch.object(
                skill, "_call_openai",
                return_value=_make_structured_llm_response(count),
            ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
                ms.SYNTHESIS_ENABLED = True
                ms.OPENAI_API_KEY = "sk-test"
                result = await skill.run(
                    update_cycle_id="15", context_pack=pack,
                    objective_count=count,
                )

            structured = result.payload.specific_objectives_structured
            assert structured[-1]["group"] == "foreign_language_it", (
                f"count={count}: last group is {structured[-1]['group']}"
            )


# ══════════════════════════════════════════════════════════════════════
# 4. Status bug fix
# ══════════════════════════════════════════════════════════════════════


class TestStatusBugFix:
    @pytest.mark.asyncio
    async def test_generated_when_structured_but_no_proposed(self):
        """If general_objective and specific_objectives are present but
        proposed_objectives is empty, status must still be 'generated'."""
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        llm_data = {
            "general_objective": "Đào tạo kỹ sư CNTT chất lượng.",
            "specific_objectives": [
                {"code": "M1", "group": "knowledge", "text": "KT cơ bản"},
                {"code": "M2", "group": "knowledge", "text": "KT chuyên ngành"},
                {"code": "M3", "group": "knowledge", "text": "KT nâng cao"},
                {"code": "M4", "group": "skills_attitude", "text": "KN nghề"},
                {"code": "M5", "group": "skills_attitude", "text": "KN mềm"},
                {"code": "M6", "group": "foreign_language_it", "text": "NN/TH"},
            ],
            "evidence_quality": "moderate",
            "warnings": [],
            # No proposed_objectives, no current_objective_analysis
        }

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        assert result.status == "generated"
        assert result.payload.general_objective == "Đào tạo kỹ sư CNTT chất lượng."
        assert len(result.payload.specific_objectives_structured) == 6


# ══════════════════════════════════════════════════════════════════════
# 5. Weak evidence warning
# ══════════════════════════════════════════════════════════════════════


class TestWeakEvidence:
    @pytest.mark.asyncio
    async def test_weak_evidence_returns_warning_not_crash(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        llm_data = {
            "general_objective": "Đào tạo sinh viên CNTT.",
            "specific_objectives": [
                {"code": "M1", "group": "knowledge", "text": "KT1"},
                {"code": "M2", "group": "knowledge", "text": "KT2"},
                {"code": "M3", "group": "knowledge", "text": "KT3"},
                {"code": "M4", "group": "skills_attitude", "text": "KN1"},
                {"code": "M5", "group": "skills_attitude", "text": "KN2"},
                {"code": "M6", "group": "foreign_language_it", "text": "NN"},
            ],
            "evidence_quality": "weak",
            "warnings": ["Nguồn minh chứng yếu, cần rà soát."],
        }

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        assert result.status == "generated"
        assert any("rà soát" in w for w in result.warnings)


# ══════════════════════════════════════════════════════════════════════
# 6. Request backward compatibility
# ══════════════════════════════════════════════════════════════════════


class TestRequestBackwardCompat:
    def test_request_without_objective_count(self):
        from app.api.v1.ctdt import ObjectiveDraftRequest

        req = ObjectiveDraftRequest(update_cycle_id="15")
        assert req.objective_count == 6  # default

    def test_request_with_objective_count(self):
        from app.api.v1.ctdt import ObjectiveDraftRequest

        req = ObjectiveDraftRequest(update_cycle_id="15", objective_count=5)
        assert req.objective_count == 5


# ══════════════════════════════════════════════════════════════════════
# 7. Response has R6.5C fields
# ══════════════════════════════════════════════════════════════════════


class TestResponseR65CFields:
    @pytest.mark.asyncio
    async def test_response_has_new_fields(self):
        from app.api.v1.ctdt import (
            ObjectiveDraftRequest,
            generate_objectives_draft,
        )
        from app.services.ctdt_objective_update_service import (
            ObjectiveDraftResult, ObjectiveSourceSummary,
        )

        mock_result = ObjectiveDraftResult(
            update_cycle_id="15", program_code="7480201", program_name="CNTT",
            draft_type="objective_update", draft_id=None, draft_saved=False,
            payload={"proposed_objectives": []},
            context_pack_summary={"role_coverage": {}, "missing_information": []},
            source_summary=ObjectiveSourceSummary(
                contexts_count=3, documents_used=[1],
                tasks_executed=["objective_update"], latency_ms=50,
            ),
            generation_status="generated", warnings=[],
            specific_objectives=[
                {"code": "M1", "group": "knowledge", "text": "KT1"},
            ],
            objective_count=5,
            format_profile="tay_nguyen_ctdt_dynamic_m_objectives",
            quality_level="good",
        )

        with patch(
            "app.services.ctdt_objective_update_service.generate_objective_update_draft",
            return_value=mock_result,
        ):
            response = await generate_objectives_draft(
                body=ObjectiveDraftRequest(update_cycle_id="15", objective_count=5),
                request=AsyncMock(), db=AsyncMock(),
                user=SimpleNamespace(tenant_id="t1", id=7),
                query_svc=AsyncMock(),
            )

        # Old fields still present
        assert hasattr(response, "payload")
        assert hasattr(response, "generation_status")
        # New R6.5C fields
        assert response.objective_count == 5
        assert response.format_profile == "tay_nguyen_ctdt_dynamic_m_objectives"
        assert hasattr(response, "quality_level")
        assert hasattr(response, "quality_messages")
        assert hasattr(response, "specific_objective_texts")
        assert hasattr(response, "general_objective_text")


# ══════════════════════════════════════════════════════════════════════
# 8. Legacy fields still populated
# ══════════════════════════════════════════════════════════════════════


class TestLegacyFieldsPopulated:
    @pytest.mark.asyncio
    async def test_specific_objective_texts_populated(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai",
            return_value=_make_structured_llm_response(6),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
                objective_count=6,
            )

        # Legacy flat texts should be populated from structured data
        assert len(result.payload.specific_objective_texts) == 6
        assert result.payload.specific_objective_texts[0].startswith("M1.")
        # general_objective_text should mirror general_objective
        assert result.payload.general_objective_text == result.payload.general_objective

    @pytest.mark.asyncio
    async def test_format_profile_set(self):
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai",
            return_value=_make_structured_llm_response(6),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        assert result.payload.format_profile == "tay_nguyen_ctdt_dynamic_m_objectives"


# ══════════════════════════════════════════════════════════════════════
# 9. R6.5C-HARDEN-1: Weak evidence default warning
# ══════════════════════════════════════════════════════════════════════


class TestHarden1WeakEvidenceDefaultWarning:
    @pytest.mark.asyncio
    async def test_weak_evidence_empty_warnings_gets_default(self):
        """evidence_quality='weak' + warnings=[] → must inject default warning."""
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        llm_data = {
            "general_objective": "Đào tạo kỹ sư CNTT.",
            "specific_objectives": [
                {"code": "M1", "group": "knowledge", "text": "KT1"},
                {"code": "M2", "group": "knowledge", "text": "KT2"},
                {"code": "M3", "group": "knowledge", "text": "KT3"},
                {"code": "M4", "group": "skills_attitude", "text": "KN1"},
                {"code": "M5", "group": "skills_attitude", "text": "KN2"},
                {"code": "M6", "group": "foreign_language_it", "text": "NN"},
            ],
            "evidence_quality": "weak",
            "warnings": [],  # empty!
        }

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        assert result.status == "generated"
        assert result.payload.evidence_quality == "weak"
        # Must have the default warning
        assert any(
            "minh chứng" in w and "rà soát" in w
            for w in result.warnings
        )

    @pytest.mark.asyncio
    async def test_weak_evidence_existing_warning_no_duplicate(self):
        """If LLM already returned a similar warning, don't duplicate."""
        from app.services.ctdt_skills.objective_update_skill import (
            ObjectiveUpdateSkill,
        )

        llm_data = {
            "general_objective": "Đào tạo kỹ sư CNTT.",
            "specific_objectives": [
                {"code": "M1", "group": "knowledge", "text": "KT1"},
                {"code": "M2", "group": "knowledge", "text": "KT2"},
                {"code": "M3", "group": "knowledge", "text": "KT3"},
                {"code": "M4", "group": "skills_attitude", "text": "KN1"},
                {"code": "M5", "group": "skills_attitude", "text": "KN2"},
                {"code": "M6", "group": "foreign_language_it", "text": "NN"},
            ],
            "evidence_quality": "weak",
            "warnings": [
                "Nguồn minh chứng yếu, cần rà soát thủ công thêm."
            ],
        }

        skill = ObjectiveUpdateSkill()
        pack = _make_context_pack()

        with patch.object(
            skill, "_call_openai", return_value=json.dumps(llm_data),
        ), patch("app.services.ctdt_skills.objective_update_skill.settings") as ms:
            ms.SYNTHESIS_ENABLED = True
            ms.OPENAI_API_KEY = "sk-test"
            result = await skill.run(
                update_cycle_id="15", context_pack=pack,
            )

        # Count warnings containing "minh chứng" + "rà soát"
        matching = [
            w for w in result.warnings
            if "minh chứng" in w and "rà soát" in w
        ]
        # Should have exactly 1 (LLM's), not 2 (no duplicate)
        assert len(matching) == 1


# ══════════════════════════════════════════════════════════════════════
# 10. R6.5C-HARDEN-1: Service quality_level with weak evidence
# ══════════════════════════════════════════════════════════════════════


class TestHarden1ServiceQualityLevel:
    def test_quality_level_warning_when_weak(self):
        """Adapter payload with evidence_quality='weak' → quality_level='warning'."""
        from app.services.ctdt_objective_quality_service import (
            adapt_objective_payload,
        )

        payload = {
            "general_objective": "Đào tạo kỹ sư CNTT.",
            "specific_objectives_structured": [
                {"code": "M1", "group": "knowledge", "text": "KT1"},
                {"code": "M2", "group": "knowledge", "text": "KT2"},
                {"code": "M3", "group": "knowledge", "text": "KT3"},
                {"code": "M4", "group": "skills_attitude", "text": "KN1"},
                {"code": "M5", "group": "skills_attitude", "text": "KN2"},
                {"code": "M6", "group": "foreign_language_it", "text": "NN"},
            ],
            "evidence_quality": "weak",
            "objective_count": 6,
            "format_profile": "tay_nguyen_ctdt_dynamic_m_objectives",
        }
        # The adapter stores evidence_quality in source_summary
        result = adapt_objective_payload(
            payload=payload, generation_status="generated",
        )
        # evidence_quality should be passed through via source_summary
        assert result.source_summary.get("objective_count") == 6
        assert result.source_summary.get("format_profile") == "tay_nguyen_ctdt_dynamic_m_objectives"


# ══════════════════════════════════════════════════════════════════════
# 11. R6.5C-HARDEN-1: Latest endpoint with _flat returns metadata
# ══════════════════════════════════════════════════════════════════════


class TestStrictObjectiveServiceQuality:
    @pytest.mark.asyncio
    async def test_service_needs_generation_has_not_attempted_message(self):
        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.NEEDS_GENERATION,
            payload=ObjectiveUpdatePayload(),
            warnings=["OPENAI_API_KEY chưa được cấu hình."],
        )

        result = await _run_service_with_skill_result(
            skill_result,
            objective_count=8,
        )

        assert result.quality_level == "failed"
        assert any(
            "Chưa thể sinh mục tiêu đào tạo" in message
            for message in result.quality_messages
        )
        assert "OPENAI_API_KEY chưa được cấu hình." in result.quality_messages
        assert not any(
            "AI trả về 0 mục tiêu" in message
            for message in result.quality_messages
        )
        assert not any(
            "AI không trả về cấu trúc mục tiêu M1 đến" in message
            for message in result.quality_messages
        )

    @pytest.mark.asyncio
    async def test_service_insufficient_context_has_not_attempted_message(self):
        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.INSUFFICIENT_CONTEXT,
            payload=ObjectiveUpdatePayload(),
            warnings=["Không có contexts nào trong context pack."],
        )

        result = await _run_service_with_skill_result(
            skill_result,
            objective_count=8,
        )

        assert result.quality_level == "failed"
        assert any(
            "Chưa đủ dữ liệu đầu vào để sinh mục tiêu đào tạo." in message
            for message in result.quality_messages
        )
        assert "Không có contexts nào trong context pack." in result.quality_messages
        assert not any(
            "AI trả về 0 mục tiêu" in message
            for message in result.quality_messages
        )

    @pytest.mark.asyncio
    async def test_service_skill_failed_quality_failed(self):
        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.FAILED,
            payload=_make_objective_payload_for_service(
                objective_count=8,
                objectives=_make_structured_objectives(5),
            ),
            warnings=["AI chỉ sinh được 5/8 mục tiêu cụ thể hợp lệ."],
        )

        result = await _run_service_with_skill_result(
            skill_result,
            objective_count=8,
        )

        assert result.quality_level == "failed"
        assert any(
            "AI trả về 5 mục tiêu trong khi yêu cầu là 8" in message
            for message in result.quality_messages
        )

    @pytest.mark.asyncio
    async def test_service_weak_evidence_valid_contract_warning_not_failed(self):
        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.GENERATED,
            payload=_make_objective_payload_for_service(
                objective_count=8,
                evidence_quality="weak",
            ),
            warnings=[],
        )

        result = await _run_service_with_skill_result(
            skill_result,
            objective_count=8,
        )

        assert result.quality_level == "warning"
        assert any("minh chứng" in message for message in result.quality_messages)

    @pytest.mark.asyncio
    async def test_service_full_count_wrong_group_uses_structural_message(self):
        objectives = _make_structured_objectives(8)
        objectives[4]["group"] = "knowledge"
        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.FAILED,
            payload=_make_objective_payload_for_service(
                objective_count=8,
                objectives=objectives,
            ),
            warnings=["M5 phải thuộc nhóm skills_attitude nhưng AI trả về knowledge."],
        )

        result = await _run_service_with_skill_result(
            skill_result,
            objective_count=8,
        )

        assert result.quality_level == "failed"
        assert any(
            "Kết quả AI không đáp ứng cấu trúc mục tiêu bắt buộc" in message
            for message in result.quality_messages
        )
        assert any(
            "M5" in message and "skills_attitude" in message
            for message in result.quality_messages
        )
        assert not any(
            "AI trả về 8 mục tiêu trong khi yêu cầu là 8" in message
            for message in result.quality_messages
        )

    @pytest.mark.asyncio
    async def test_service_count_mismatch_backstop_failed(self):
        skill_result = ObjectiveUpdateResult(
            status=ObjectiveUpdateStatus.GENERATED,
            payload=_make_objective_payload_for_service(
                objective_count=8,
                objectives=_make_structured_objectives(5),
            ),
            warnings=[],
        )

        result = await _run_service_with_skill_result(
            skill_result,
            objective_count=8,
        )

        assert result.quality_level == "failed"


class TestHarden1LatestEndpointMetadata:
    @pytest.mark.asyncio
    async def test_latest_with_flat_has_metadata(self):
        """GET latest with _flat block returns all R6.5C metadata."""
        from app.api.v1.ctdt import get_latest_objective_draft

        flat_data = {
            "general_objective": "Mục tiêu chung.",
            "general_objective_text": "Mục tiêu chung.",
            "specific_objectives": [
                {"code": "M1", "group": "knowledge", "text": "KT1"},
            ],
            "specific_objective_texts": ["M1. KT1"],
            "objective_count": 5,
            "format_profile": "tay_nguyen_ctdt_dynamic_m_objectives",
            "quality_level": "warning",
            "quality_messages": ["Thiếu minh chứng."],
            "evidence_quality": "weak",
            "source_summary": {},
            "warnings": ["Thiếu minh chứng."],
        }

        mock_draft = MagicMock()
        mock_draft.id = 42
        mock_draft.status = "draft"
        mock_draft.program_name = "CNTT"
        mock_draft.program_code = "7480201"
        mock_draft.result_payload = {"_flat": flat_data}
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

        data = response.data
        assert data["objective_count"] == 5
        assert data["format_profile"] == "tay_nguyen_ctdt_dynamic_m_objectives"
        assert data["quality_level"] == "warning"
        assert data["quality_messages"] == ["Thiếu minh chứng."]
        assert data["general_objective_text"] == "Mục tiêu chung."
        assert data["specific_objective_texts"] == ["M1. KT1"]
        assert data["evidence_quality"] == "weak"


# ══════════════════════════════════════════════════════════════════════
# 12. R6.5C-HARDEN-1: Old draft fallback safe defaults
# ══════════════════════════════════════════════════════════════════════


class TestHarden1OldDraftFallback:
    @pytest.mark.asyncio
    async def test_old_draft_no_flat_has_safe_defaults(self):
        """Old draft without _flat → R6.5C fields get safe defaults."""
        from app.api.v1.ctdt import get_latest_objective_draft

        mock_draft = MagicMock()
        mock_draft.id = 10
        mock_draft.status = "draft"
        mock_draft.program_name = "CNTT"
        mock_draft.program_code = "7480201"
        mock_draft.result_payload = {
            "proposed_objectives": [
                {"objective_type": "general_objective",
                 "proposed_content": "Đào tạo kỹ sư CNTT."},
                {"objective_type": "specific_objective",
                 "proposed_content": "Kiến thức cơ sở."},
            ],
            "_meta": {"generation_status": "generated"},
            # no _flat!
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

        data = response.data
        # Must not crash
        assert data["draft_id"] == 10
        # Safe defaults
        assert data["objective_count"] == 6
        assert data["format_profile"] == ""
        assert data["quality_level"] in ("good", "warning")
        assert isinstance(data["quality_messages"], list)
        assert data["general_objective_text"] == data["general_objective"]
        assert isinstance(data["specific_objective_texts"], list)
        assert data["evidence_quality"] == "moderate"
