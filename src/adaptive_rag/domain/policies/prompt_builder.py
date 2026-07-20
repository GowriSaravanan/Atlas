"""Load prompt templates and build chat messages for answer generation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from adaptive_rag.config.settings import AnswerGenerationSettings


class PromptBuilder:
    """Construct provider-agnostic chat messages from external templates."""

    def __init__(self, settings: AnswerGenerationSettings) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        self._prompts_dir = repo_root / settings.prompts_dir

    def build_messages(self, query: str, context: str) -> list[dict[str, str]]:
        """Return system + user messages for answer generation."""
        system_prompt = self._load_template("system.txt")
        user_prompt = self._load_template("answer_generation.txt").format(
            query=query.strip(),
            context=context.strip(),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @lru_cache(maxsize=4)
    def _load_template(self, filename: str) -> str:
        path = self._prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Prompt template not found: {path}")
        return path.read_text(encoding="utf-8").strip()
