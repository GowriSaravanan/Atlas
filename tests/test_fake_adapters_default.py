"""Verify unit tests continue to use fake AI adapters."""

from __future__ import annotations

from adaptive_rag.api.dependencies.container import get_container
from adaptive_rag.infrastructure.embeddings.fake_embedder import FakeEmbedder
from adaptive_rag.infrastructure.llm.fake_llm import FakeLLM
from adaptive_rag.infrastructure.reranking.fake_reranker import FakeReranker


def test_unit_tests_use_fake_adapters() -> None:
    container = get_container()
    assert isinstance(container.embedder, FakeEmbedder)
    assert isinstance(container.reranker, FakeReranker)
    assert isinstance(container.llm, FakeLLM)
