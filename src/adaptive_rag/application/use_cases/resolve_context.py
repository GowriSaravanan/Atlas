"""Resolve conversation context for a query."""

from __future__ import annotations

from adaptive_rag.config.settings import Settings
from adaptive_rag.domain.models.conversation import Message


class ResolveConversationContextUseCase:
    """Load and prepare conversation context window."""

    def __init__(self, settings: Settings) -> None:
        self._max_turns = settings.conversation.max_turns
        self._summary_threshold = settings.conversation.summary_threshold

    def execute(
        self,
        *,
        conversation_id: str,
        messages: list[Message],
    ) -> tuple[list[Message], str | None]:
        """Return the context window and optional summary of older turns."""
        _ = conversation_id  # reserved for persistence in Phase 1+
        if len(messages) <= self._max_turns:
            return messages, None
        return messages[-self._max_turns :], None
