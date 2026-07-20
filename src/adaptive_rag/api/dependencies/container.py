"""Dependency injection composition root."""

from __future__ import annotations

from dataclasses import dataclass, field

from adaptive_rag.application.use_cases.ingest_document import IngestDocumentUseCase
from adaptive_rag.application.use_cases.query_rag import QueryRAGUseCase
from adaptive_rag.application.use_cases.resolve_context import ResolveConversationContextUseCase
from adaptive_rag.application.workflow.ingest_pipeline import compile_ingest_graph
from adaptive_rag.application.workflow.query_graph import compile_query_graph
from adaptive_rag.config.settings import Settings, get_settings


@dataclass
class Container:
    """Application dependency container — wires use cases and ports."""

    settings: Settings = field(default_factory=get_settings)

    # Compiled graphs (singletons)
    _query_graph: object | None = field(default=None, repr=False)
    _ingest_graph: object | None = field(default=None, repr=False)

    # Use cases
    _query_rag_use_case: QueryRAGUseCase | None = field(default=None, repr=False)
    _ingest_document_use_case: IngestDocumentUseCase | None = field(default=None, repr=False)
    _resolve_context_use_case: ResolveConversationContextUseCase | None = field(
        default=None, repr=False
    )

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
            self._ingest_graph = compile_ingest_graph()
        return self._ingest_graph

    @property
    def query_rag_use_case(self) -> QueryRAGUseCase:
        """Return query RAG use case."""
        if self._query_rag_use_case is None:
            self._query_rag_use_case = QueryRAGUseCase()
        return self._query_rag_use_case

    @property
    def ingest_document_use_case(self) -> IngestDocumentUseCase:
        """Return ingest document use case."""
        if self._ingest_document_use_case is None:
            self._ingest_document_use_case = IngestDocumentUseCase()
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
