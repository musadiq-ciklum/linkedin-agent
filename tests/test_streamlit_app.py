from app import format_context_label


def test_format_context_label_returns_expected_string():
    label = format_context_label("chunk_001", 0.87)
    assert "chunk_001" in label
    assert "0.87" in label


def test_format_context_label_rounds_score_to_two_decimal_places():
    label = format_context_label("doc_42", 0.12345)
    assert "0.12" in label
    assert "0.12345" not in label
