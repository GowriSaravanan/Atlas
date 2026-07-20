"""Deterministic LLM adapter for tests."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, Field

from adaptive_rag.config.settings import LLMSettings
from adaptive_rag.domain.ports.llm import LLMPort

T = TypeVar("T", bound=BaseModel)


class FakeLLM(LLMPort):
    """Return predictable answers derived from the prompt for unit tests."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self._settings = settings or LLMSettings()
        self.last_messages: list[dict[str, str]] | None = None
        self.last_prompt: str | None = None

    @property
    def model_name(self) -> str:
        return f"fake-{self._settings.model}"

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        del temperature
        self.last_prompt = prompt
        return self._answer_from_text(prompt)

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        del temperature
        if hasattr(schema, "model_validate"):
            payload: dict[str, Any] = {"answer": self._answer_from_text(prompt)}
            for field_name, field in schema.model_fields.items():
                if field_name not in payload:
                    if field.annotation is str:
                        payload[field_name] = ""
                    elif field.annotation is int:
                        payload[field_name] = 0
                    elif field.annotation is float:
                        payload[field_name] = 0.0
                    elif field.annotation is bool:
                        payload[field_name] = False
            return schema.model_validate(payload)
        raise TypeError("Schema must be a Pydantic model")

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        del temperature
        self.last_messages = messages
        combined = "\n".join(message["content"] for message in messages)
        return self._answer_from_text(combined)

    def get_config(self) -> dict[str, Any]:
        return {
            "provider": "fake",
            "model": self.model_name,
            "temperature": self._settings.temperature,
        }

    @staticmethod
    def _answer_from_text(prompt: str) -> str:
        if "Evidence:" not in prompt:
            return "I do not have enough evidence to answer that question."
        evidence = prompt.split("Evidence:", maxsplit=1)[1]
        evidence = evidence.split("Answer:", maxsplit=1)[0].strip()
        first_line = next((line for line in evidence.splitlines() if line.strip()), "")
        if not first_line:
            return "I do not have enough evidence to answer that question."
        return f"Based on the provided evidence: {first_line[:240]}"
