# src/prompts/prompt_builder.py
from pathlib import Path


class PromptBuilder:
    """
    A minimal, test-aligned prompt builder.
    """

    VALID_MODES = {
        "chat",
        "summarizer",
        "agent",
        "refusal",
        "rag_query",
        "rerank_query",
    }

    def __init__(self, mode="chat", prompts_dir=None, max_chars=12000):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}")

        self.mode = mode
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None
        self.max_chars = max_chars

        self.system_templates = {}
        self.user_templates = {}

        if self.prompts_dir:
            self._load_templates()

    # ---------------------------------------------------------
    # Load templates
    # ---------------------------------------------------------
    def _load_templates(self):
        system_dir = self.prompts_dir / "system"
        user_dir = self.prompts_dir / "user"

        for mode in self.VALID_MODES:
            sys_file = system_dir / f"{mode}.txt"
            user_file = user_dir / f"{mode}.txt"

            if sys_file.exists():
                self.system_templates[mode] = sys_file.read_text().strip()

            if user_file.exists():
                self.user_templates[mode] = user_file.read_text().strip()

    # ---------------------------------------------------------
    # Context formatting
    # ---------------------------------------------------------
    def format_context(self, chunks):
        """
        Public wrapper for tests.
        """
        return self._format_context(chunks)

    def _format_context(self, chunks):
        if not chunks:
            return ""

        blocks = []
        for c in chunks:
            doc_id = c.id if hasattr(c, "id") else c["id"]
            text = c.text if hasattr(c, "text") else c["text"]

            blocks.append(f'<doc id="{doc_id}">{text}</doc>')

        return "<context>\n" + "\n".join(blocks) + "\n</context>"

    def _trim_context(self, text):
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars]

    # ---------------------------------------------------------
    # Build
    # ---------------------------------------------------------
    def build(self, user_query, chunks):
        """
        Build the final prompt for the LLM.

        - Refusal ONLY when mode == "refusal"
        - Empty chunks → no <context>, but still include system + user
        """

        # 1. Explicit refusal mode
        if self.mode == "refusal":
            system_raw = self.system_templates.get("refusal", "")
            user_raw = self.user_templates.get("refusal", "")
            return system_raw + "\n\n" + user_raw.replace("{{query}}", user_query)

        # 2. Normal modes (rag_query, chat, agent, summarizer, rerank)
        system_raw = self.system_templates.get(self.mode, "")
        user_raw = self.user_templates.get(self.mode, "")

        parts = []

        if system_raw:
            parts.append(system_raw)

        if chunks:
            context = self._format_context(chunks)
            parts.append(context)

        # ALWAYS include user query
        parts.append(user_raw.replace("{{query}}", user_query))

        return "\n\n".join(parts)
