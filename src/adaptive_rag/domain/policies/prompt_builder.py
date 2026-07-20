"""Load prompt templates and build chat messages for answer generation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


class PromptBuilder:
    """Construct provider-agnostic chat messages from external templates."""

    def __init__(self, prompts_dir: Path) -> None:
        self._prompts_dir = prompts_dir

    def build_messages(self, query: str, context: str) -> list[dict[str, str]]:
        """Return system + user messages for answer generation."""
        system_prompt = _load_template(str(self._prompts_dir.resolve()), "system.txt")
        user_prompt = _load_template(str(self._prompts_dir.resolve()), "answer_generation.txt").format(
            query=query.strip(),
            context=context.strip(),
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]


@lru_cache(maxsize=8)
def _load_template(prompts_dir: str, filename: str) -> str:
    path = Path(prompts_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8").strip()
