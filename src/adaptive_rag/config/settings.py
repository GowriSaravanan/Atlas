"""Application configuration via Pydantic Settings."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
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


class LLMSettings(BaseSettings):
    """LLM provider configuration."""

    provider: Literal["openai", "gemini", "groq", "ollama"] = "ollama"
    model: str = "llama3"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=256, le=16384)
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = Field(default=32, ge=1, le=256)


class RerankerSettings(BaseSettings):
    """Cross-encoder reranker configuration."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
    eval_mode: bool = False

    conversation: ConversationSettings = Field(default_factory=ConversationSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    confidence_weights: ConfidenceWeightSettings = Field(default_factory=ConfidenceWeightSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    sparse_index: SparseIndexSettings = Field(default_factory=SparseIndexSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
