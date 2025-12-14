# tests/test_prompts.py
import pytest
from pathlib import Path
from src.prompts.prompt_builder import PromptBuilder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build(mode="chat", prompts_dir=None, max_chars=12000):
    return PromptBuilder(mode=mode, prompts_dir=prompts_dir, max_chars=max_chars)

TEST_CHUNKS = [
    {"id": "c1", "text": "Alpha"},
    {"id": "c2", "text": "Beta"},
]

# ---------------------------------------------------------------------------
# Test: invalid mode
# ---------------------------------------------------------------------------
def test_invalid_mode():
    with pytest.raises(ValueError):
        build(mode="invalid")

# ---------------------------------------------------------------------------
# Test: refusal mode
# ---------------------------------------------------------------------------
def test_refusal_mode(tmp_path):
    # create minimal template files
    system_file = tmp_path / "system" / "refusal.txt"
    user_file = tmp_path / "user" / "refusal.txt"
    system_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.parent.mkdir(parents=True, exist_ok=True)
    system_file.write_text("SYSTEM_REFUSAL")
    user_file.write_text("I cannot assist with {{query}}.")

    pb = build(mode="refusal", prompts_dir=tmp_path)
    out = pb.build("hack?", [])
    assert "I cannot assist" in out
    assert "SYSTEM_REFUSAL" in out

# ---------------------------------------------------------------------------
# Test: RAG query
# ---------------------------------------------------------------------------
def test_rag_query_multiple_chunks(tmp_path):
    system_file = tmp_path / "system" / "rag_query.txt"
    user_file = tmp_path / "user" / "rag_query.txt"
    system_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.parent.mkdir(parents=True, exist_ok=True)
    system_file.write_text("RAG_SYSTEM")
    user_file.write_text("Question: {{query}}")

    pb = build(mode="rag_query", prompts_dir=tmp_path)
    out = pb.build("What is this?", TEST_CHUNKS)

    assert "<context>" in out
    assert "<doc id=\"c1\">Alpha</doc>" in out
    assert "<doc id=\"c2\">Beta</doc>" in out
    assert "What is this?" in out
    assert "RAG_SYSTEM" in out

def test_rag_query_empty_chunks(tmp_path):
    # create minimal template files so build() can work
    system_file = tmp_path / "system" / "rag_query.txt"
    user_file = tmp_path / "user" / "rag_query.txt"
    system_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.parent.mkdir(parents=True, exist_ok=True)
    system_file.write_text("RAG_SYSTEM")
    user_file.write_text("Question: {{query}}")

    pb = build(mode="rag_query", prompts_dir=tmp_path)
    out = pb.build("Explain", [])
    # context should be empty, but query should appear
    assert "<context>" not in out
    assert "Explain" in out
    assert "RAG_SYSTEM" in out

# ---------------------------------------------------------------------------
# Test: Rerank query
# ---------------------------------------------------------------------------
def test_rerank_query(tmp_path):
    system_file = tmp_path / "system" / "rerank_query.txt"
    user_file = tmp_path / "user" / "rerank_query.txt"
    system_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.parent.mkdir(parents=True, exist_ok=True)
    system_file.write_text("RERANK_SYSTEM")
    user_file.write_text("Sort: {{query}}")

    pb = build(mode="rerank_query", prompts_dir=tmp_path)
    # pass at least one valid chunk
    out = pb.build("Sort these", [{"id": "c1", "text": "Dummy"}])
    assert out.startswith("RERANK_SYSTEM")
    assert "Sort these" in out

# ---------------------------------------------------------------------------
# Test: context formatting
# ---------------------------------------------------------------------------
def test_format_context_single_block():
    pb = build()
    chunks = [{"id": "x", "text": "Hello"}]
    out = pb.format_context(chunks)
    assert "<doc id=\"x\">Hello</doc>" in out
    assert "<context>" in out

def test_format_context_whitespace():
    pb = build()
    chunks = [{"id": "x", "text": " " * 10}]
    out = pb.format_context(chunks)
    assert out.strip() != ""

# ---------------------------------------------------------------------------
# Test: load templates
# ---------------------------------------------------------------------------
def test_load_templates(tmp_path):
    system_file = tmp_path / "system" / "chat.txt"
    user_file = tmp_path / "user" / "chat.txt"
    system_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.parent.mkdir(parents=True, exist_ok=True)
    system_file.write_text("Chat system instruction.")
    user_file.write_text("Chat user instruction: {{query}}")

    pb = build(mode="chat", prompts_dir=tmp_path)
    assert "Chat system instruction." in pb.system_templates.get("chat", "")
    assert "Chat user instruction" in pb.user_templates.get("chat", "")

# ---------------------------------------------------------------------------
# Test: trim context
# ---------------------------------------------------------------------------
def test_trim_context():
    pb = build(max_chars=10)
    text = "ABCDEFGHIJKL"
    trimmed = pb._trim_context(text)
    assert len(trimmed) <= 10

# ---------------------------------------------------------------------------
# Test: chat mode
# ---------------------------------------------------------------------------
def test_chat_mode(tmp_path):
    system_file = tmp_path / "system" / "chat.txt"
    user_file = tmp_path / "user" / "chat.txt"
    system_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.parent.mkdir(parents=True, exist_ok=True)
    system_file.write_text("CHAT_SYSTEM")
    user_file.write_text("Chat query: {{query}}")

    pb = build(mode="chat", prompts_dir=tmp_path)
    out = pb.build("What is this?", TEST_CHUNKS)
    assert "CHAT_SYSTEM" in out
    assert "<context>" in out
    assert "What is this?" in out

# ---------------------------------------------------------------------------
# Test: summarizer mode
# ---------------------------------------------------------------------------
def test_summarizer_mode(tmp_path):
    system_file = tmp_path / "system" / "summarizer.txt"
    user_file = tmp_path / "user" / "summarizer.txt"
    system_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.parent.mkdir(parents=True, exist_ok=True)
    system_file.write_text("SUM_SYSTEM")
    user_file.write_text("Summarize: {{query}}")

    pb = build(mode="summarizer", prompts_dir=tmp_path)
    out = pb.build("Summarize it.", TEST_CHUNKS)
    assert "SUM_SYSTEM" in out
    assert "<context>" in out
    assert "Summarize it." in out

# ---------------------------------------------------------------------------
# Test: agent mode
# ---------------------------------------------------------------------------
def test_agent_mode(tmp_path):
    system_file = tmp_path / "system" / "agent.txt"
    user_file = tmp_path / "user" / "agent.txt"
    system_file.parent.mkdir(parents=True, exist_ok=True)
    user_file.parent.mkdir(parents=True, exist_ok=True)
    system_file.write_text("AGENT_SYSTEM")
    user_file.write_text("Plan: {{query}}")

    pb = build(mode="agent", prompts_dir=tmp_path)
    out = pb.build("Plan this task.", TEST_CHUNKS)
    assert "AGENT_SYSTEM" in out
    assert "<context>" in out
    assert "Plan this task." in out
