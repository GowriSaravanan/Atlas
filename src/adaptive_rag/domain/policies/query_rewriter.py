"""Rule-based query rewriting policy."""

from __future__ import annotations

import re

from adaptive_rag.domain.models.conversation import Message, MessageRole
from adaptive_rag.domain.models.query import QueryAnalysis, RewriteResult
from adaptive_rag.domain.ports.query_rewriter import QueryRewriterPort

_FOLLOWUP_PREFIX_PATTERN = re.compile(
    r"^\s*(what about|how about|and what about|tell me about)\s+(?P<topic>.+?)\??\s*$",
    re.IGNORECASE,
)
_TRAILING_FILLER_PATTERN = re.compile(r"\b(please|thanks|thank you)\.?\s*$", re.IGNORECASE)


class RuleBasedQueryRewriter(QueryRewriterPort):
    """Deterministic query rewriter for follow-ups and ambiguous phrasing."""

    def rewrite(
        self,
        query: str,
        analysis: QueryAnalysis,
        *,
        context_messages: list[Message] | None = None,
    ) -> RewriteResult:
        """Rewrite the query when analysis indicates it is required."""
        original = query.strip()
        if not analysis.rewrite_decision.should_rewrite:
            return RewriteResult(
                original_query=original,
                rewritten_query=original,
                was_rewritten=False,
                reason=analysis.rewrite_decision.reason or "rewrite not required",
            )

        rewritten = self._rewrite_with_rules(original, analysis, context_messages or [])
        if rewritten != original:
            return RewriteResult(
                original_query=original,
                rewritten_query=rewritten,
                was_rewritten=True,
                reason=analysis.rewrite_decision.reason,
            )

        return RewriteResult(
            original_query=original,
            rewritten_query=original,
            was_rewritten=False,
            reason="rewrite requested but no safe rule-based rewrite applied",
        )

    def _rewrite_with_rules(
        self,
        query: str,
        analysis: QueryAnalysis,
        context_messages: list[Message],
    ) -> str:
        text = _TRAILING_FILLER_PATTERN.sub("", query).strip()
        match = _FOLLOWUP_PREFIX_PATTERN.match(text)
        if match:
            topic = match.group("topic").strip(" .")
            topic = re.sub(r"^(the|a|an)\s+", "", topic, flags=re.IGNORECASE).strip()
            topic = self._enrich_topic_from_context(topic, context_messages)
            return self._to_policy_question(topic, analysis)

        if analysis.metadata_scope.get("department"):
            dept = analysis.metadata_scope["department"]
            return f"What is the {dept} policy regarding {text.rstrip('?')}?"

        context_topic = self._latest_user_topic(context_messages)
        if context_topic and self._contains_pronoun(text):
            return f"{context_topic.rstrip('?')}?"

        return text

    @staticmethod
    def _to_policy_question(topic: str, analysis: QueryAnalysis) -> str:
        topic = topic.strip(" .")
        topic_lower = topic.lower()
        department_prefix = ""
        if analysis.metadata_scope.get("department"):
            department = str(analysis.metadata_scope["department"])
            if not topic_lower.startswith(department.lower()):
                department_prefix = f"{department} "

        if topic_lower.endswith(" policy") or topic_lower.endswith(" benefits"):
            return f"What is the {department_prefix}{topic}?"
        if any(keyword in topic_lower for keyword in ("policy", "leave", "benefit", "handbook")):
            return f"What is the {department_prefix}{topic} policy?"
        return f"What is {topic}?"

    @staticmethod
    def _contains_pronoun(text: str) -> bool:
        return bool(re.search(r"\b(it|this|that|they|them)\b", text, re.IGNORECASE))

    @staticmethod
    def _latest_user_topic(context_messages: list[Message]) -> str | None:
        for message in reversed(context_messages):
            if message.role != MessageRole.USER:
                continue
            content = message.content.strip()
            if content:
                return content
        return None

    @staticmethod
    def _enrich_topic_from_context(topic: str, context_messages: list[Message]) -> str:
        if not RuleBasedQueryRewriter._contains_pronoun(topic):
            return topic

        latest = RuleBasedQueryRewriter._latest_user_topic(context_messages)
        if not latest:
            return topic

        return latest.replace("?", "").strip()
