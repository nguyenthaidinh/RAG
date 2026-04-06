from app.services.document_metadata_builder import DocumentMetadataBuilder


def test_semantic_builder_returns_expected_shape():
    builder = DocumentMetadataBuilder()
    result = builder.build(
        title="Quy chế đào tạo 2026",
        text="Đây là quy chế đào tạo dành cho sinh viên và giảng viên trong năm 2026.",
        file_name="quyche.pdf",
        original_name="quyche.pdf",
        content_type="application/pdf",
        size_bytes=1234,
        ingest_via="upload",
    )

    assert "system" in result
    assert "document_identity" in result
    assert "semantic" in result
    assert "structure" in result
    assert "retrieval_hints" in result
    assert result["system"]["pipeline_mode"] == "semantic"