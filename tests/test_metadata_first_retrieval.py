from app.services.retrieval.metadata_first_retrieval import MetadataFirstRetrievalService


def test_parse_query_to_conditions_regulation():
    svc = MetadataFirstRetrievalService()
    cond = svc._parse_query_to_conditions("quy chế đào tạo mới nhất cho sinh viên")

    assert "regulation" in cond.document_kinds
    assert "đào tạo" in cond.topics
    assert "student" in cond.audience
    assert cond.freshness_sensitive is True


def test_is_good_enough():
    svc = MetadataFirstRetrievalService()

    class Dummy:
        def __init__(self, score):
            self.score = score

    assert svc.is_good_enough([Dummy(0.8)]) is True
    assert svc.is_good_enough([Dummy(0.4)]) is False
    assert svc.is_good_enough([]) is False