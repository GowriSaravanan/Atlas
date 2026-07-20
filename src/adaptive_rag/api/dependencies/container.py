"""Dependency injection composition root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from adaptive_rag.application.use_cases.hybrid_retrieval import HybridRetrievalUseCase
from adaptive_rag.application.use_cases.ingest_document import IngestDocumentUseCase
from adaptive_rag.application.use_cases.query_rag import QueryRAGUseCase
from adaptive_rag.application.use_cases.resolve_context import ResolveConversationContextUseCase
from adaptive_rag.application.workflow.ingest_pipeline import compile_ingest_graph
from adaptive_rag.application.workflow.nodes.ingest_nodes import IngestNodeContext
from adaptive_rag.application.workflow.query_graph import compile_query_graph
from adaptive_rag.config.mappers import (
    resolve_prompts_dir,
    to_answer_generation_policy_config,
    to_citation_formatter_policy_config,
    to_chunking_policy_config,
    to_confidence_weight_config,
    to_fusion_policy_config,
    to_retrieval_policy_config,
)
from adaptive_rag.config.settings import Settings, get_settings
from adaptive_rag.domain.policies.adaptive_chunker import AdaptiveChunker
from adaptive_rag.domain.policies.confidence import ConfidenceScorer
from adaptive_rag.domain.policies.document_metadata_extractor import DocumentMetadataExtractor
from adaptive_rag.domain.policies.rrf import ReciprocalRankFusion
from adaptive_rag.domain.ports.embedder import EmbedderPort

if TYPE_CHECKING:
    from adaptive_rag.domain.ports.answer_generator import AnswerGeneratorPort
    from adaptive_rag.domain.ports.citation_formatter import CitationFormatterPort
    from adaptive_rag.domain.ports.index_registry import IndexRegistryPort
    from adaptive_rag.domain.ports.llm import LLMPort
    from adaptive_rag.domain.ports.reranker import RerankerPort


def _use_fake_embedder() -> bool:
    return os.getenv("ADAPTIVE_RAG_FAKE_EMBEDDER", "").lower() in {"1", "true", "yes"}


def _use_fake_reranker() -> bool:
    return os.getenv("ADAPTIVE_RAG_FAKE_RERANKER", "").lower() in {"1", "true", "yes"}


def _use_fake_llm() -> bool:
    return os.getenv("ADAPTIVE_RAG_FAKE_LLM", "").lower() in {"1", "true", "yes"}


@dataclass
class Container:
    """Application dependency container — wires use cases and ports."""

    settings: Settings = field(default_factory=get_settings)

    _embedder: EmbedderPort | None = field(default=None, repr=False)
    _reranker: RerankerPort | None = field(default=None, repr=False)
    _llm: LLMPort | None = field(default=None, repr=False)
    _answer_generator: AnswerGeneratorPort | None = field(default=None, repr=False)
    _citation_formatter: CitationFormatterPort | None = field(default=None, repr=False)
    _fusion_engine: object | None = field(default=None, repr=False)
    _confidence_scorer: ConfidenceScorer | None = field(default=None, repr=False)
    _index_registry: IndexRegistryPort | None = field(default=None, repr=False)
    _ingest_context: IngestNodeContext | None = field(default=None, repr=False)

    # Compiled graphs (singletons)
    _query_graph: object | None = field(default=None, repr=False)
    _ingest_graph: object | None = field(default=None, repr=False)

    # Use cases
    _query_rag_use_case: QueryRAGUseCase | None = field(default=None, repr=False)
    _retrieval_engine: object | None = field(default=None, repr=False)
    _hybrid_retriever: object | None = field(default=None, repr=False)
    _hybrid_retrieval_use_case: HybridRetrievalUseCase | None = field(default=None, repr=False)
    _ingest_document_use_case: IngestDocumentUseCase | None = field(default=None, repr=False)
    _resolve_context_use_case: ResolveConversationContextUseCase | None = field(
        default=None, repr=False
    )

    def ensure_storage_dirs(self) -> None:
        """Create configured storage directories."""
        for path in (
            self.settings.storage.data_dir,
            self.settings.storage.index_dir,
            self.settings.storage.upload_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)

    @property
    def embedder(self) -> EmbedderPort:
        if self._embedder is None:
            if _use_fake_embedder():
                from adaptive_rag.infrastructure.embeddings.fake_embedder import FakeEmbedder

                self._embedder = FakeEmbedder()
            else:
                from adaptive_rag.infrastructure.embeddings.sentence_transformer import (
                    SentenceTransformerEmbedder,
                )

                self._embedder = SentenceTransformerEmbedder(self.settings.embedding)
        return self._embedder

    @property
    def reranker(self) -> RerankerPort:
        if self._reranker is None:
            if _use_fake_reranker():
                from adaptive_rag.infrastructure.reranking.fake_reranker import FakeReranker

                self._reranker = FakeReranker()
            else:
                from adaptive_rag.infrastructure.reranking.cross_encoder import CrossEncoderReranker

                self._reranker = CrossEncoderReranker(self.settings.reranker)
        return self._reranker

    @property
    def llm(self) -> LLMPort:
        if self._llm is None:
            from adaptive_rag.infrastructure.llm import build_llm

            self._llm = build_llm(self.settings.llm)
        return self._llm

    @property
    def answer_generator(self) -> AnswerGeneratorPort:
        if self._answer_generator is None:
            from adaptive_rag.domain.policies.context_builder import ContextBuilder
            from adaptive_rag.domain.policies.prompt_builder import PromptBuilder
            from adaptive_rag.infrastructure.llm.fake_answer_generator import FakeAnswerGenerator
            from adaptive_rag.infrastructure.llm.llm_answer_generator import LLMAnswerGenerator

            context_builder = ContextBuilder(
                to_answer_generation_policy_config(self.settings.answer_generation)
            )
            prompt_builder = PromptBuilder(resolve_prompts_dir(self.settings))
            if _use_fake_llm():
                self._answer_generator = FakeAnswerGenerator(
                    context_builder=context_builder,
                    prompt_builder=prompt_builder,
                )
            else:
                self._answer_generator = LLMAnswerGenerator(
                    llm=self.llm,
                    context_builder=context_builder,
                    prompt_builder=prompt_builder,
                )
        return self._answer_generator

    @property
    def citation_formatter(self) -> CitationFormatterPort:
        if self._citation_formatter is None:
            from adaptive_rag.domain.policies.evidence_citation_formatter import (
                EvidenceCitationFormatter,
            )

            self._citation_formatter = EvidenceCitationFormatter(
                to_citation_formatter_policy_config(self.settings.citation)
            )
        return self._citation_formatter

    @property
    def index_registry(self) -> IndexRegistryPort:
        if self._index_registry is None:
            from adaptive_rag.infrastructure.factories import (
                build_sparse_retriever_factory,
                build_vector_store_factory,
            )
            from adaptive_rag.infrastructure.storage.collection_index import CollectionIndexRegistry

            self.ensure_storage_dirs()
            self._index_registry = CollectionIndexRegistry(
                base_path=Path(self.settings.storage.index_dir),
                embedder=self.embedder,
                vector_store_factory=build_vector_store_factory(self.settings),
                sparse_retriever_factory=build_sparse_retriever_factory(self.settings),
                vector_backend=self.settings.vector_store.provider,
                sparse_backend=self.settings.sparse_index.provider,
            )
        return self._index_registry

    @property
    def ingest_context(self) -> IngestNodeContext:
        if self._ingest_context is None:
            from adaptive_rag.infrastructure.pdf.pymupdf_loader import PyMuPDFLoader

            self._ingest_context = IngestNodeContext(
                document_loader=PyMuPDFLoader(),
                metadata_extractor=DocumentMetadataExtractor(),
                chunker=AdaptiveChunker(to_chunking_policy_config(self.settings.chunking)),
                index_registry=self.index_registry,
            )
        return self._ingest_context

    @property
    def query_graph(self):
        """Return compiled query workflow graph."""
        if self._query_graph is None:
            self._query_graph = compile_query_graph()
        return self._query_graph

    @property
    def ingest_graph(self):
        """Return compiled ingest workflow graph."""
        if self._ingest_graph is None:
            self._ingest_graph = compile_ingest_graph(self.ingest_context)
        return self._ingest_graph

    @property
    def query_rag_use_case(self) -> QueryRAGUseCase:
        """Return query RAG use case."""
        if self._query_rag_use_case is None:
            self._query_rag_use_case = QueryRAGUseCase()
        return self._query_rag_use_case

    @property
    def fusion_engine(self):
        if self._fusion_engine is None:
            self._fusion_engine = ReciprocalRankFusion(
                to_fusion_policy_config(self.settings.retrieval)
            )
        return self._fusion_engine

    @property
    def confidence_scorer(self) -> ConfidenceScorer:
        if self._confidence_scorer is None:
            self._confidence_scorer = ConfidenceScorer(
                retrieval=to_retrieval_policy_config(self.settings.retrieval),
                weights=to_confidence_weight_config(self.settings.confidence_weights),
            )
        return self._confidence_scorer

    @property
    def hybrid_retriever(self):
        if self._hybrid_retriever is None:
            from adaptive_rag.application.services.hybrid_retriever import HybridRetriever

            self._hybrid_retriever = HybridRetriever(
                index_registry=self.index_registry,
                fusion_engine=self.fusion_engine,
                settings=self.settings.retrieval,
            )
        return self._hybrid_retriever

    @property
    def retrieval_engine(self):
        if self._retrieval_engine is None:
            from adaptive_rag.application.services.retrieval_engine import RetrievalEngine

            self._retrieval_engine = RetrievalEngine(
                index_registry=self.index_registry,
                hybrid_retriever=self.hybrid_retriever,
                settings=self.settings.retrieval,
                fusion_engine=self.fusion_engine,
                confidence_scorer=self.confidence_scorer,
                reranker=self.reranker,
                answer_generator=self.answer_generator,
                citation_formatter=self.citation_formatter,
            )
        return self._retrieval_engine

    @property
    def hybrid_retrieval_use_case(self) -> HybridRetrievalUseCase:
        if self._hybrid_retrieval_use_case is None:
            self._hybrid_retrieval_use_case = HybridRetrievalUseCase(self.retrieval_engine)
        return self._hybrid_retrieval_use_case

    @property
    def ingest_document_use_case(self) -> IngestDocumentUseCase:
        """Return ingest document use case."""
        if self._ingest_document_use_case is None:
            self._ingest_document_use_case = IngestDocumentUseCase(self.ingest_graph)
        return self._ingest_document_use_case

    @property
    def resolve_context_use_case(self) -> ResolveConversationContextUseCase:
        """Return resolve context use case."""
        if self._resolve_context_use_case is None:
            self._resolve_context_use_case = ResolveConversationContextUseCase(self.settings)
        return self._resolve_context_use_case


_container: Container | None = None


def get_container() -> Container:
    """Return the global application container (lazy singleton)."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container() -> None:
    """Reset container — for testing only."""
    global _container
    _container = None
