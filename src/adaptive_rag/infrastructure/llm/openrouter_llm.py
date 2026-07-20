"""OpenRouter LLM adapter."""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from adaptive_rag.config.settings import LLMSettings
from adaptive_rag.domain.errors import ProviderError
from adaptive_rag.domain.ports.llm import LLMPort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProviderLLM(LLMPort):
    """OpenAI-compatible chat completions via OpenRouter."""

    def __init__(self, settings: LLMSettings) -> None:
        self._settings = settings

    @property
    def model_name(self) -> str:
        return self._settings.model

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        return self.generate_messages(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        temperature: float = 0.0,
    ) -> T:
        raw = self.generate(prompt, temperature=temperature)
        if hasattr(schema, "model_validate_json"):
            return schema.model_validate_json(raw)
        raise TypeError("Schema must support JSON validation")

    def generate_messages(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        if not self._settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter provider")

        payload = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._settings.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/GowriSaravanan/Atlas-RAG",
            "X-Title": self._settings.openrouter_app_title,
        }
        try:
            response = httpx.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter request failed: {exc}") from exc

        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()

    def get_config(self) -> dict[str, Any]:
        return {
            "provider": "openrouter",
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }
