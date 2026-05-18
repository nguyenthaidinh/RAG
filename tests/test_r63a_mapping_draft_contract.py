"""Tests for R6.3A Mapping Draft Contract."""
from __future__ import annotations
import inspect
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest


# ══ 1. normalize_contribution_level(None) → 0 ══
class TestContribNone:
    def test_none_returns_zero(self):
        from app.services.ctdt_mapping_draft_contract import normalize_contribution_level
        level, warnings = normalize_contribution_level(None)
        assert level == 0
        assert warnings == []


# ══ 2. normalize_contribution_level("") → 0 ══
class TestContribEmpty:
    def test_empty_returns_zero(self):
        from app.services.ctdt_mapping_draft_contract import normalize_contribution_level
        level, warnings = normalize_contribution_level("")
        assert level == 0
        assert warnings == []


# ══ 3. normalize_contribution_level("1"/"2"/"3") → int ══
class TestContribStrings:
    @pytest.mark.parametrize("val,expected", [("1", 1), ("2", 2), ("3", 3), ("0", 0)])
    def test_string_digits(self, val, expected):
        from app.services.ctdt_mapping_draft_contract import normalize_contribution_level
        level, warnings = normalize_contribution_level(val)
        assert level == expected
        assert warnings == []

    @pytest.mark.parametrize("val,expected", [(1, 1), (2, 2), (3, 3), (0, 0)])
    def test_int_values(self, val, expected):
        from app.services.ctdt_mapping_draft_contract import normalize_contribution_level
        level, warnings = normalize_contribution_level(val)
        assert level == expected
        assert warnings == []


# ══ 4. normalize_contribution_level("X") → 1 + warning ══
class TestContribX:
    @pytest.mark.parametrize("val", ["X", "x"])
    def test_x_maps_to_low(self, val):
        from app.services.ctdt_mapping_draft_contract import normalize_contribution_level
        level, warnings = normalize_contribution_level(val)
        assert level == 1
        assert len(warnings) == 1
        assert "X" in warnings[0] or "x" in warnings[0]


# ══ 5. Invalid contribution → 0 + warning ══
class TestContribInvalid:
    @pytest.mark.parametrize("val", [5, -1, "abc", "high", True])
    def test_invalid_returns_zero_with_warning(self, val):
        from app.services.ctdt_mapping_draft_contract import normalize_contribution_level
        level, warnings = normalize_contribution_level(val)
        assert level == 0
        assert len(warnings) == 1


# ══ 6. ObjectiveOutcomeMappingRow serializable ══
class TestObjOutcomeRowSerialize:
    def test_asdict(self):
        from app.services.ctdt_mapping_draft_contract import (
            ObjectiveOutcomeMappingRow, MappingSourceRef,
        )
        row = ObjectiveOutcomeMappingRow(
            objective_code="PO1", outcome_code="PLO1",
            contribution_level=2, confidence="high",
            source_refs=[MappingSourceRef(ai_document_id=1)],
        )
        d = asdict(row)
        assert d["objective_code"] == "PO1"
        assert d["outcome_code"] == "PLO1"
        assert d["contribution_level"] == 2
        assert len(d["source_refs"]) == 1
        assert d["source_refs"][0]["ai_document_id"] == 1


# ══ 7. CourseOutcomeMappingRow serializable ══
class TestCourseOutcomeRowSerialize:
    def test_asdict(self):
        from app.services.ctdt_mapping_draft_contract import CourseOutcomeMappingRow
        row = CourseOutcomeMappingRow(
            course_code="CS101", course_name="AI cơ bản",
            outcome_code="PLO1", contribution_level=3,
        )
        d = asdict(row)
        assert d["course_code"] == "CS101"
        assert d["contribution_level"] == 3
        assert isinstance(d["source_refs"], list)


# ══ 8. CLOProgramOutcomeMappingRow serializable ══
class TestCLORowSerialize:
    def test_asdict(self):
        from app.services.ctdt_mapping_draft_contract import (
            CourseLearningOutcomeProgramOutcomeMappingRow,
        )
        row = CourseLearningOutcomeProgramOutcomeMappingRow(
            course_code="CS101", course_outcome_code="H1",
            program_outcome_code="PLO1", contribution_level=1,
        )
        d = asdict(row)
        assert d["course_outcome_code"] == "H1"
        assert d["program_outcome_code"] == "PLO1"


# ══ 9. MappingDraftPayload contains all 3 row groups ══
class TestPayloadComplete:
    def test_payload_all_groups(self):
        from app.services.ctdt_mapping_draft_contract import (
            MappingDraftPayload, ObjectiveOutcomeMappingRow,
            CourseOutcomeMappingRow,
            CourseLearningOutcomeProgramOutcomeMappingRow,
            MappingSourceRef,
        )
        payload = MappingDraftPayload(
            update_cycle_id="15", program_code="7480201",
            objective_outcome_rows=[
                ObjectiveOutcomeMappingRow(objective_code="PO1", outcome_code="PLO1",
                    contribution_level=2,
                    source_refs=[MappingSourceRef(ai_document_id=10)]),
            ],
            course_outcome_rows=[
                CourseOutcomeMappingRow(course_code="CS101", outcome_code="PLO1"),
            ],
            clo_program_outcome_rows=[
                CourseLearningOutcomeProgramOutcomeMappingRow(
                    course_code="CS101", course_outcome_code="H1",
                    program_outcome_code="PLO1"),
            ],
        )
        d = payload.to_dict()
        assert len(d["objective_outcome_rows"]) == 1
        assert len(d["course_outcome_rows"]) == 1
        assert len(d["clo_program_outcome_rows"]) == 1

        summary = payload.build_source_summary()
        assert 10 in summary["documents_used"]
        assert summary["rows_count"]["objective_outcome"] == 1
        assert summary["rows_count"]["course_outcome"] == 1
        assert summary["rows_count"]["course_learning_outcome_program_outcome"] == 1


# ══ 10. source_refs preserves metadata ══
class TestSourceRefMetadata:
    def test_source_ref_fields(self):
        from app.services.ctdt_mapping_draft_contract import MappingSourceRef
        ref = MappingSourceRef(
            ai_document_id=5, update_cycle_id="15",
            program_code="7480201", program_id="p1",
        )
        d = asdict(ref)
        assert d["update_cycle_id"] == "15"
        assert d["program_code"] == "7480201"
        assert d["program_id"] == "p1"


# ══ 11. dedupe_warnings ══
class TestDedupeWarnings:
    def test_deduplicates(self):
        from app.services.ctdt_mapping_draft_contract import dedupe_warnings
        result = dedupe_warnings(["a", "b", "a", "c", "b"])
        assert result == ["a", "b", "c"]

    def test_empty(self):
        from app.services.ctdt_mapping_draft_contract import dedupe_warnings
        assert dedupe_warnings([]) == []


# ══ 12. No LLM/httpx/OpenAI imports ══
class TestNoLLMImports:
    def test_no_llm_imports(self):
        import app.services.ctdt_mapping_draft_contract as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "openai" not in joined.lower()
        assert "httpx" not in joined.lower()
        assert "SYNTHESIS" not in joined


# ══ 13. No Program model imports ══
class TestNoProgramImports:
    def test_no_program_imports(self):
        import app.services.ctdt_mapping_draft_contract as mod
        src = inspect.getsource(mod)
        imports = [l.strip() for l in src.splitlines() if l.strip().startswith(("import ", "from "))]
        joined = "\n".join(imports)
        assert "ProgramVersion" not in joined
        assert "from app.db.models.program" not in joined


# ══ 14. normalize_confidence ══
class TestNormalizeConfidence:
    @pytest.mark.parametrize("val,expected", [
        ("low", "low"), ("HIGH", "high"), ("", "medium"), (None, "medium"), ("xyz", "medium"),
    ])
    def test_normalize(self, val, expected):
        from app.services.ctdt_mapping_draft_contract import normalize_confidence
        assert normalize_confidence(val) == expected


# ══ 15. normalize_mapping_status ══
class TestNormalizeMappingStatus:
    @pytest.mark.parametrize("val,expected", [
        ("draft", "draft"), ("needs_review", "needs_review"),
        ("user_confirmed", "user_confirmed"),
        ("", "draft"), (None, "draft"), ("approved", "draft"),
    ])
    def test_normalize(self, val, expected):
        from app.services.ctdt_mapping_draft_contract import normalize_mapping_status
        assert normalize_mapping_status(val) == expected


# ══ 16. normalize_source_type ══
class TestNormalizeSourceType:
    @pytest.mark.parametrize("val,expected", [
        ("extracted_from_current_curriculum", "extracted_from_current_curriculum"),
        ("generated_from_draft", "generated_from_draft"),
        ("", "unknown"), (None, "unknown"), ("manual", "unknown"),
    ])
    def test_normalize(self, val, expected):
        from app.services.ctdt_mapping_draft_contract import normalize_source_type
        assert normalize_source_type(val) == expected


# ══ 17. Schema endpoint returns contract info ══
class TestSchemaEndpoint:
    @pytest.mark.asyncio
    async def test_schema_endpoint(self):
        from app.api.v1.ctdt import mapping_draft_schema
        resp = await mapping_draft_schema(
            request=AsyncMock(),
            user=SimpleNamespace(tenant_id="t1", id=7),
        )
        assert resp["draft_type"] == "mapping_draft"
        assert "objective_outcome" in resp["mapping_types"]
        assert "course_outcome" in resp["mapping_types"]
        assert resp["contribution_levels"]["0"] == "no_contribution"
        assert resp["contribution_levels"]["3"] == "high"


# ══ 18. default_factory no shared state ══
class TestDefaultFactoryIsolation:
    def test_no_shared_mutable(self):
        from app.services.ctdt_mapping_draft_contract import (
            ObjectiveOutcomeMappingRow, CourseOutcomeMappingRow,
            MappingDraftPayload,
        )
        r1 = ObjectiveOutcomeMappingRow()
        r2 = ObjectiveOutcomeMappingRow()
        r1.warnings.append("test")
        assert r2.warnings == []

        p1 = MappingDraftPayload()
        p2 = MappingDraftPayload()
        p1.objective_outcome_rows.append(r1)
        assert p2.objective_outcome_rows == []


# ══ 19. rows_count uses canonical keys, no legacy ══
class TestRowsCountCanonicalKeys:
    def test_no_legacy_clo_key(self):
        from app.services.ctdt_mapping_draft_contract import MappingDraftPayload
        payload = MappingDraftPayload(update_cycle_id="15")
        summary = payload.build_source_summary()
        assert "clo_program_outcome" not in summary["rows_count"]
        assert set(summary["rows_count"].keys()) == {
            "objective_outcome",
            "course_outcome",
            "course_learning_outcome_program_outcome",
        }


# ══ 20. rows_by_mapping_type keys match MappingDraftType.ALL ══
class TestRowsByMappingType:
    def test_keys_match_all(self):
        from app.services.ctdt_mapping_draft_contract import (
            MappingDraftPayload, MappingDraftType,
            ObjectiveOutcomeMappingRow,
            CourseLearningOutcomeProgramOutcomeMappingRow,
        )
        payload = MappingDraftPayload(
            update_cycle_id="15",
            objective_outcome_rows=[ObjectiveOutcomeMappingRow(objective_code="PO1")],
            clo_program_outcome_rows=[
                CourseLearningOutcomeProgramOutcomeMappingRow(course_code="CS101"),
            ],
        )
        by_type = payload.rows_by_mapping_type()
        assert set(by_type.keys()) == MappingDraftType.ALL
        assert len(by_type[MappingDraftType.OBJECTIVE_OUTCOME]) == 1
        assert len(by_type[MappingDraftType.COURSE_LEARNING_OUTCOME_PROGRAM_OUTCOME]) == 1
        assert len(by_type[MappingDraftType.COURSE_OUTCOME]) == 0


# ══ 21. Schema endpoint has CLO mapping type ══
class TestSchemaEndpointCLO:
    @pytest.mark.asyncio
    async def test_schema_has_clo_type(self):
        from app.api.v1.ctdt import mapping_draft_schema
        resp = await mapping_draft_schema(
            request=AsyncMock(),
            user=SimpleNamespace(tenant_id="t1", id=7),
        )
        assert "course_learning_outcome_program_outcome" in resp["mapping_types"]
        assert len(resp["mapping_types"]) == 3
