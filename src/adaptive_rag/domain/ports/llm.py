"""LLM port."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMPort(Protocol):
    """Language model abstraction decoupled from provider SDKs."""

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""
        ...

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Generate unstructured text from a prompt."""
        ...

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        """Generate structured output conforming to a Pydantic schema."""
        ...

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        """Generate from a chat message list."""
        ...

    def get_config(self) -> dict[str, Any]:
        """Return provider configuration metadata for tracing."""
        ...
