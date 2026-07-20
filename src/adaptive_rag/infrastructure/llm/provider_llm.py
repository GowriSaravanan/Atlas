"""Provider-backed LLM adapters."""

from __future__ import annotations

from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from adaptive_rag.config.settings import LLMSettings
from adaptive_rag.domain.ports.llm import LLMPort
from adaptive_rag.observability.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ProviderLLM(LLMPort):
    """HTTP-backed LLM adapter for supported providers."""

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
        provider = self._settings.provider
        if provider == "ollama":
            return self._generate_ollama(messages, temperature=temperature)
        if provider == "openai":
            return self._generate_openai_compatible(
                base_url="https://api.openai.com/v1",
                api_key=self._settings.openai_api_key,
                messages=messages,
                temperature=temperature,
            )
        if provider == "groq":
            return self._generate_openai_compatible(
                base_url="https://api.groq.com/openai/v1",
                api_key=self._settings.groq_api_key,
                messages=messages,
                temperature=temperature,
            )
        if provider == "gemini":
            return self._generate_gemini(messages, temperature=temperature)
        raise ValueError(f"Unsupported LLM provider: {provider}")

    def get_config(self) -> dict[str, Any]:
        return {
            "provider": self._settings.provider,
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
        }

    def _generate_ollama(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
    ) -> str:
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        url = f"{self._settings.ollama_base_url.rstrip('/')}/api/chat"
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        return str(data["message"]["content"]).strip()

    def _generate_openai_compatible(
        self,
        *,
        base_url: str,
        api_key: str | None,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        if not api_key:
            raise ValueError("API key is required for the selected LLM provider")
        payload = {
            "model": self._settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self._settings.max_tokens,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        response = httpx.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()

    def _generate_gemini(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
    ) -> str:
        if not self._settings.gemini_api_key:
            raise ValueError("GEMINI API key is required for Gemini provider")
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        user_parts = [message["content"] for message in messages if message["role"] != "system"]
        prompt = "\n\n".join(system_parts + user_parts)
        model = self._settings.model
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self._settings.gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        return str(data["candidates"][0]["content"]["parts"][0]["text"]).strip()
