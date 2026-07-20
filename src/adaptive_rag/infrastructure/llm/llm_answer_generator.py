"""LLM-backed answer generator."""

from __future__ import annotations

import time

from adaptive_rag.domain.models.answer import GeneratedAnswer
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.policies.context_builder import ContextBuilder
from adaptive_rag.domain.policies.prompt_builder import PromptBuilder
from adaptive_rag.domain.policies.token_utils import estimate_token_count
from adaptive_rag.domain.ports.answer_generator import AnswerGeneratorPort
from adaptive_rag.domain.ports.llm import LLMPort


class LLMAnswerGenerator(AnswerGeneratorPort):
    """Generate grounded answers using a configurable LLM provider."""

    def __init__(
        self,
        *,
        llm: LLMPort,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
    ) -> None:
        self._llm = llm
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder

    def generate(self, query: str, evidence: list[ScoredChunk]) -> GeneratedAnswer:
        start = time.perf_counter()
        built = self._context_builder.build(evidence)
        messages = self._prompt_builder.build_messages(query, built.context)
        prompt_text = "\n".join(message["content"] for message in messages)

        if not built.used_chunks:
            answer = "I do not have enough evidence to answer that question."
        else:
            answer = self._llm.generate_messages(messages).strip() or (
                "I do not have enough evidence to answer that question."
            )

        prompt_tokens = estimate_token_count(prompt_text)
        completion_tokens = estimate_token_count(answer)
        latency_ms = (time.perf_counter() - start) * 1000

        return GeneratedAnswer(
            answer=answer,
            used_chunk_ids=built.used_chunk_ids,
            model_name=self._llm.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 2),
        )
