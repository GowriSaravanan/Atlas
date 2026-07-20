"""Prompt builder unit tests."""

from __future__ import annotations

from adaptive_rag.config.settings import AnswerGenerationSettings
from adaptive_rag.domain.policies.prompt_builder import PromptBuilder


def test_prompt_builder_loads_templates_and_formats_messages() -> None:
    builder = PromptBuilder(AnswerGenerationSettings())
    messages = builder.build_messages(
        query="How many sick leave days?",
        context="[Evidence 1 | chunk_id=c1 | section=Sick Leave | policy_id=HR-105]\n10 days",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "provided evidence" in messages[0]["content"].lower()
    assert messages[1]["role"] == "user"
    assert "How many sick leave days?" in messages[1]["content"]
    assert "10 days" in messages[1]["content"]
    assert "Answer:" in messages[1]["content"]
