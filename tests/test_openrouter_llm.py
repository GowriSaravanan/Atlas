"""OpenRouter LLM unit tests."""

from __future__ import annotations

import httpx
import pytest

from adaptive_rag.config.settings import LLMSettings
from adaptive_rag.infrastructure.llm.openrouter_llm import OpenRouterProviderLLM


def test_openrouter_generate_messages_uses_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_post(url: str, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["json"] = kwargs.get("json")
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Ten sick leave days."}}]},
            request=request,
        )

    monkeypatch.setattr(httpx, "post", _fake_post)
    llm = OpenRouterProviderLLM(
        LLMSettings(
            provider="openrouter",
            model="meta-llama/llama-3.1-8b-instruct",
            openrouter_api_key="test-key",
        )
    )

    answer = llm.generate_messages(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "How many sick days?"},
        ]
    )

    assert answer == "Ten sick leave days."
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    headers = captured["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    payload = captured["json"]
    assert payload["model"] == "meta-llama/llama-3.1-8b-instruct"


def test_openrouter_requires_api_key() -> None:
    llm = OpenRouterProviderLLM(
        LLMSettings(provider="openrouter", model="meta-llama/llama-3.1-8b-instruct")
    )
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        llm.generate("hello")
