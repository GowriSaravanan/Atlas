"""Application configuration via Pydantic Settings."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from adaptive_rag.domain.models.index import SparseBackend, VectorBackend


class ConversationSettings(BaseSettings):
    """Conversation context configuration."""

    max_turns: int = Field(default=5, ge=1, le=50)
    summary_threshold: int = Field(default=10, ge=2, le=200)


class RetrievalSettings(BaseSettings):
    """Retrieval pipeline configuration."""

    default_top_k: int = Field(default=10, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    max_escalations: int = Field(default=3, ge=1, le=5)
    rrf_k: int = Field(default=60, ge=1)


class ConfidenceWeightSettings(BaseSettings):
    """Weights for composite confidence scoring."""

    reranker_score: float = 0.30
    reranker_margin: float = 0.20
    retrieval_overlap: float = 0.20
    metadata_match: float = 0.15
    evidence_density: float = 0.15


class AnswerGenerationSettings(BaseSettings):
    """Answer generation configuration."""

    max_context_tokens: int = Field(default=2048, ge=256, le=16384)
    prompts_dir: str = "prompts"


class CitationSettings(BaseSettings):
    """Citation formatting configuration."""

    excerpt_max_chars: int = Field(default=200, ge=50, le=2000)


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    provider: Literal["openai", "gemini", "groq", "ollama", "openrouter"] = "openrouter"
    model: str = "meta-llama/llama-3.1-8b-instruct"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=256, le=16384)
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    openrouter_app_title: str = "Adaptive Hybrid RAG Platform"
    ollama_base_url: str = "http://localhost:11434"


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_name: str = "BAAI/bge-base-en-v1.5"
    batch_size: int = Field(default=32, ge=1, le=256)


class RerankerSettings(BaseSettings):
    """Cross-encoder reranker configuration."""

    model_name: str = "BAAI/bge-reranker-base"
    batch_size: int = Field(default=16, ge=1, le=128)


class VectorStoreSettings(BaseSettings):
    """Vector store provider configuration."""

    provider: VectorBackend = VectorBackend.FAISS
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None


class SparseIndexSettings(BaseSettings):
    """Sparse index provider configuration."""

    provider: SparseBackend = SparseBackend.BM25


class ChunkingSettings(BaseSettings):
    """Adaptive chunking configuration."""

    max_tokens: int = Field(default=512, ge=64, le=4096)
    min_tokens: int = Field(default=64, ge=16, le=1024)
    overlap_tokens: int = Field(default=50, ge=0, le=512)
    strategy: Literal["structure_aware", "fixed"] = "structure_aware"


class StorageSettings(BaseSettings):
    """Local storage paths for indices and uploads."""

    data_dir: str = "data"
    index_dir: str = "data/indices"
    upload_dir: str = "data/uploads"
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)


class LoggingSettings(BaseSettings):
    """Structured logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    json_logs: bool = False
    service_name: str = "adaptive-rag"


class Settings(BaseSettings):
    """Root application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    app_name: str = "Adaptive Hybrid RAG Platform"
    app_version: str = "0.1.0"
    debug: bool = False

    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    confidence_weights: ConfidenceWeightSettings = Field(default_factory=ConfidenceWeightSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    sparse_index: SparseIndexSettings = Field(default_factory=SparseIndexSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    answer_generation: AnswerGenerationSettings = Field(default_factory=AnswerGenerationSettings)
    citation: CitationSettings = Field(default_factory=CitationSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="before")
    @classmethod
    def apply_flat_env_aliases(cls, data: Any) -> Any:
        """Map top-level production env vars onto nested settings."""
        payload = dict(data or {})

        def lookup(name: str) -> str | None:
            env_value = os.getenv(name)
            if env_value:
                return env_value
            raw = payload.get(name)
            return raw if isinstance(raw, str) and raw else None

        embedding = dict(payload.get("embedding") or {})
        if embedding_model := lookup("EMBEDDING_MODEL"):
            embedding["model_name"] = embedding_model
        if embedding:
            payload["embedding"] = embedding

        reranker = dict(payload.get("reranker") or {})
        if reranker_model := lookup("RERANKER_MODEL"):
            reranker["model_name"] = reranker_model
        if reranker:
            payload["reranker"] = reranker

        llm = dict(payload.get("llm") or {})
        if openrouter_api_key := lookup("OPENROUTER_API_KEY"):
            llm["openrouter_api_key"] = openrouter_api_key
        if openrouter_model := lookup("OPENROUTER_MODEL"):
            llm["model"] = openrouter_model
            llm["provider"] = "openrouter"
        if llm:
            payload["llm"] = llm

        return payload


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
