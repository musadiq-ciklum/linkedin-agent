# scripts/prompt_builder_test.py
from pathlib import Path
from src.prompts.prompt_builder import PromptBuilder

ROOT = Path(__file__).parent.parent
PROMPTS_DIR = ROOT / "prompts_test"

# ---------------------------------------------------------
# Setup minimal prompt templates on disk
# ---------------------------------------------------------
(PROMPTS_DIR / "system").mkdir(parents=True, exist_ok=True)
(PROMPTS_DIR / "user").mkdir(parents=True, exist_ok=True)

(PROMPTS_DIR / "system" / "chat.txt").write_text(
    "CHAT SYSTEM INSTRUCTION"
)
(PROMPTS_DIR / "user" / "chat.txt").write_text(
    "Answer the question: {{query}}"
)

(PROMPTS_DIR / "system" / "rag_query.txt").write_text(
    "RAG SYSTEM INSTRUCTION"
)
(PROMPTS_DIR / "user" / "rag_query.txt").write_text(
    "Question: {{query}}"
)

(PROMPTS_DIR / "system" / "refusal.txt").write_text(
    "REFUSAL SYSTEM"
)
(PROMPTS_DIR / "user" / "refusal.txt").write_text(
    "I cannot help with {{query}}."
)

# ---------------------------------------------------------
# Test data
# ---------------------------------------------------------
chunks = [
    {"id": "c1", "text": "Alpha chunk"},
    {"id": "c2", "text": "Beta chunk"},
]

# ---------------------------------------------------------
# CHAT MODE
# ---------------------------------------------------------
print("\n==== CHAT MODE ====\n")
pb = PromptBuilder(mode="chat", prompts_dir=PROMPTS_DIR)
out = pb.build("What is this?", chunks)
print(out)

# ---------------------------------------------------------
# RAG QUERY MODE
# ---------------------------------------------------------
print("\n==== RAG QUERY MODE ====\n")
pb = PromptBuilder(mode="rag_query", prompts_dir=PROMPTS_DIR)
out = pb.build("Explain this", chunks)
print(out)

# ---------------------------------------------------------
# REFUSAL MODE
# ---------------------------------------------------------
print("\n==== REFUSAL MODE ====\n")
pb = PromptBuilder(mode="refusal", prompts_dir=PROMPTS_DIR)
out = pb.build("hack a server", [])
print(out)
