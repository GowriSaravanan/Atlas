"""Build LLM-ready context blocks from reranked evidence."""

from __future__ import annotations

from adaptive_rag.config.settings import AnswerGenerationSettings
from adaptive_rag.domain.models.answer import BuiltContext
from adaptive_rag.domain.models.retrieval import ScoredChunk
from adaptive_rag.domain.policies.token_utils import estimate_token_count


class ContextBuilder:
    """Select, order, and truncate reranked chunks into a context block."""

    def __init__(self, settings: AnswerGenerationSettings) -> None:
        self._max_context_tokens = settings.max_context_tokens

    def build(self, evidence: list[ScoredChunk]) -> BuiltContext:
        """Build a token-budgeted context block preserving chunk metadata."""
        if not evidence:
            return BuiltContext(context="", used_chunks=[], used_chunk_ids=[], estimated_tokens=0)

        ordered = sorted(
            evidence,
            key=lambda hit: hit.rank if hit.rank is not None else 10_000,
        )
        blocks: list[str] = []
        used_chunks: list[ScoredChunk] = []
        total_tokens = 0
        truncated = False

        for index, hit in enumerate(ordered, start=1):
            block = self._format_chunk(index, hit)
            block_tokens = estimate_token_count(block)
            if used_chunks and total_tokens + block_tokens > self._max_context_tokens:
                truncated = True
                break
            if not used_chunks and block_tokens > self._max_context_tokens:
                block = self._truncate_block(block, self._max_context_tokens)
                block_tokens = estimate_token_count(block)
                truncated = True

            blocks.append(block)
            used_chunks.append(hit)
            total_tokens += block_tokens

            if truncated:
                break

        context = "\n\n".join(blocks)
        return BuiltContext(
            context=context,
            used_chunks=used_chunks,
            used_chunk_ids=[hit.chunk.id for hit in used_chunks],
            estimated_tokens=total_tokens,
            truncated=truncated,
        )

    @staticmethod
    def _format_chunk(index: int, hit: ScoredChunk) -> str:
        metadata = hit.chunk.metadata
        section = metadata.get("section_title", "unknown")
        policy_id = metadata.get("policy_id", "unknown")
        header = (
            f"[Evidence {index} | chunk_id={hit.chunk.id} | "
            f"section={section} | policy_id={policy_id}]"
        )
        return f"{header}\n{hit.chunk.content.strip()}"

    @staticmethod
    def _truncate_block(block: str, max_tokens: int) -> str:
        words = block.split()
        if not words:
            return block
        max_words = max(1, int(max_tokens / 1.3))
        if len(words) <= max_words:
            return block
        return " ".join(words[:max_words]) + " ..."
