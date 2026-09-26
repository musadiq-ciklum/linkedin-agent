from unittest.mock import MagicMock, patch

from app import format_context_label, generate_session_title


def test_format_context_label_returns_expected_string():
    label = format_context_label("chunk_001", 0.87)
    assert "chunk_001" in label
    assert "0.87" in label


def test_format_context_label_rounds_score_to_two_decimal_places():
    label = format_context_label("doc_42", 0.12345)
    assert "0.12" in label
    assert "0.12345" not in label


# ── generate_session_title ────────────────────────────────────────────────────

def test_generate_session_title_returns_llm_generated_title():
    mock_pipeline = MagicMock()
    mock_pipeline.llm_client.generate.return_value = MagicMock(text="  Authentication Setup Guide  ")

    with patch("app.get_pipeline", return_value=mock_pipeline):
        title = generate_session_title("How do I set up authentication?")

    assert title == "Authentication Setup Guide"


def test_generate_session_title_strips_whitespace_from_llm_response():
    mock_pipeline = MagicMock()
    mock_pipeline.llm_client.generate.return_value = MagicMock(text="\n  Machine Learning Basics \n")

    with patch("app.get_pipeline", return_value=mock_pipeline):
        title = generate_session_title("What is machine learning?")

    assert title == "Machine Learning Basics"


def test_generate_session_title_passes_first_message_to_llm():
    mock_pipeline = MagicMock()
    mock_pipeline.llm_client.generate.return_value = MagicMock(text="Some Title")

    with patch("app.get_pipeline", return_value=mock_pipeline):
        generate_session_title("Tell me about ChromaDB")

    call_args = mock_pipeline.llm_client.generate.call_args[0][0]
    assert "Tell me about ChromaDB" in call_args
