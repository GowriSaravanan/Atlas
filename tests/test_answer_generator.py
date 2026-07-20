"""Answer generator unit and integration tests."""

from __future__ import annotations

from adaptive_rag.config.settings import AnswerGenerationSettings, LLMSettings
from adaptive_rag.domain.models.document import Chunk
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.policies.context_builder import ContextBuilder
from adaptive_rag.domain.policies.prompt_builder import PromptBuilder
from adaptive_rag.infrastructure.llm.fake_llm import FakeLLM
from adaptive_rag.infrastructure.llm.llm_answer_generator import LLMAnswerGenerator


def _evidence() -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=Chunk(
                id="c1",
                document_id="doc-1",
                content="Employees may take up to 10 sick leave days per year.",
                metadata={"section_title": "Sick Leave", "policy_id": "HR-105"},
            ),
            score=0.91,
            source="reranker",
            rank=1,
        )
    ]


def test_llm_answer_generator_uses_context_and_returns_metadata() -> None:
    settings = AnswerGenerationSettings()
    llm = FakeLLM(LLMSettings())
    generator = LLMAnswerGenerator(
        llm=llm,
        context_builder=ContextBuilder(settings),
        prompt_builder=PromptBuilder(settings),
    )

    result = generator.generate("How many sick leave days?", _evidence())

    assert result.answer
    assert result.used_chunk_ids == ["c1"]
    assert result.model_name.startswith("fake-")
    assert result.prompt_tokens > 0
    assert result.completion_tokens > 0
    assert result.latency_ms >= 0.0
    assert llm.last_messages is not None
    assert "10 sick leave days" in llm.last_messages[1]["content"]


def test_retrieval_engine_returns_generated_answer(sample_pdf) -> None:
    from adaptive_rag.api.dependencies.container import get_container

    container = get_container()
    container.ensure_storage_dirs()
    container.ingest_document_use_case.execute(source_path=str(sample_pdf), collection_id="default")

    result = container.hybrid_retrieval_use_case.execute(
        query="How many sick leave days?",
        collection_id="default",
        top_k=5,
    )

    assert result.generated_answer is not None
    assert result.generated_answer.answer
    assert result.generated_answer.used_chunk_ids
    assert "sick leave" in result.generated_answer.answer.lower()
    assert any(step.step == "answer_generation" for step in result.trace.steps)
