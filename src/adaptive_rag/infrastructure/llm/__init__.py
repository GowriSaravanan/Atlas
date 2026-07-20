"""LLM infrastructure adapters."""

from __future__ import annotations

import os

from adaptive_rag.config.settings import LLMSettings
from adaptive_rag.domain.ports.llm import LLMPort
from adaptive_rag.infrastructure.llm.fake_llm import FakeLLM
from adaptive_rag.infrastructure.llm.openrouter_llm import OpenRouterProviderLLM
from adaptive_rag.infrastructure.llm.provider_llm import ProviderLLM


def _use_fake_llm() -> bool:
    return os.getenv("ADAPTIVE_RAG_FAKE_LLM", "").lower() in {"1", "true", "yes"}


def build_llm(settings: LLMSettings) -> LLMPort:
    """Construct an LLM adapter from settings and environment toggles."""
    if _use_fake_llm():
        return FakeLLM(settings)
    if settings.provider == "openrouter":
        return OpenRouterProviderLLM(settings)
    return ProviderLLM(settings)
