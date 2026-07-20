"""Deterministic answer generator for tests and eval."""

from __future__ import annotations

import time

from adaptive_rag.domain.models.answer import GeneratedAnswer
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.policies.context_builder import ContextBuilder
from adaptive_rag.domain.policies.prompt_builder import PromptBuilder
from adaptive_rag.domain.policies.token_utils import estimate_token_count
from adaptive_rag.domain.ports.answer_generator import AnswerGeneratorPort


class FakeAnswerGenerator(AnswerGeneratorPort):
    """Generate grounded answers from retrieved evidence without calling an LLM."""

    def __init__(
        self,
        *,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        model_name: str = "fake-answer-generator",
    ) -> None:
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._model_name = model_name

    def generate(self, query: str, evidence: list[ScoredChunk]) -> GeneratedAnswer:
        start = time.perf_counter()
        built = self._context_builder.build(evidence)
        messages = self._prompt_builder.build_messages(query, built.context)
        prompt_text = "\n".join(message["content"] for message in messages)

        if not built.used_chunks:
            answer = "I do not have enough evidence to answer that question."
        else:
            top = built.used_chunks[0].chunk.content.strip()
            answer = f"Based on the provided evidence: {top[:400]}"

        prompt_tokens = estimate_token_count(prompt_text)
        completion_tokens = estimate_token_count(answer)
        latency_ms = (time.perf_counter() - start) * 1000

        return GeneratedAnswer(
            answer=answer,
            used_chunk_ids=built.used_chunk_ids,
            model_name=self._model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 2),
        )
