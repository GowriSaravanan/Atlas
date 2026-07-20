"""Rule-based query analysis policy."""

from __future__ import annotations

import re

from adaptive_rag.domain.models.query import (
    ComplexityLevel,
    DecompositionDecision,
    QueryAnalysis,
    QueryIntent,
    QueryType,
    RewriteDecision,
)

_POLICY_ID_PATTERN = re.compile(r"\b([A-Z]{2,10}-\d{2,6})\b")
_DEPARTMENT_PATTERN = re.compile(
    r"\b(HR|Finance|Legal|Engineering|Operations|IT|Sales|Marketing)\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
_VERSION_PATTERN = re.compile(r"\b(v\d+(?:\.\d+)?|version\s+\d+)\b", re.IGNORECASE)
_MULTI_QUESTION_PATTERN = re.compile(r"\?")
_GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|good morning|good afternoon|thanks|thank you)\b",
    re.IGNORECASE,
)
_COMPARATIVE_PATTERN = re.compile(
    r"\b(compare|versus|vs\.?|difference between|better|worse)\b",
    re.IGNORECASE,
)
_PROCEDURAL_PATTERN = re.compile(r"\b(how to|how do i|steps|procedure|process)\b", re.IGNORECASE)
_SUMMARIZE_PATTERN = re.compile(r"\b(summarize|summary|overview|tl;dr)\b", re.IGNORECASE)
_REWRITE_FOLLOWUP_PATTERN = re.compile(
    r"^\s*(what about|how about|and what about|tell me about)\b",
    re.IGNORECASE,
)
_VAGUE_PRONOUN_PATTERN = re.compile(r"\b(it|this|that|they|them)\b", re.IGNORECASE)
_DISTINCT_COMPARISON_PATTERN = re.compile(
    r"(\bcompare\b.+\b(versus|vs\.?|with|and)\b|\bdifference between\b.+\band\b|^\s*both\b.+\band\b|^\s*(?:versus|vs\.?)\b.+\band\b)",
    re.IGNORECASE,
)
_SAME_TOPIC_COMPOUND_PATTERN = re.compile(
    r"\b(policy|leave|benefits?|handbook|procedure)\b.*\band\b.*\b(how many|carry over|apply|eligible|work)\b",
    re.IGNORECASE,
)
_BOTH_AND_PATTERN = re.compile(r"\bboth\b.+\band\b", re.IGNORECASE)
_POLICY_DIFFERENCES_PATTERN = re.compile(
    r"\b(policy|leave)\b.+\band\b.+\b(policy|leave)\b.*\bdifferences?\b",
    re.IGNORECASE,
)
_HR_DOMAIN_PATTERN = re.compile(
    r"\b(leave|benefit|policy|handbook|employee|maternity|paternity|sick|annual)\b",
    re.IGNORECASE,
)


class QueryAnalyzer:
    """Hybrid query analyzer — rules first; LLM enrichment in Phase 4+."""

    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze a query using deterministic rules."""
        text = query.strip()
        rule_matches: list[str] = []
        entities: list[str] = []
        metadata_hints: dict[str, str | int] = {}
        metadata_scope: dict[str, str | int] = {}

        if _GREETING_PATTERN.search(text):
            rule_matches.append("greeting")
            rewrite = RewriteDecision(should_rewrite=False, reason="conversational greeting")
            decompose = DecompositionDecision(should_decompose=False, reason="conversational greeting")
            return self._build_analysis(
                query_type=QueryType.CONVERSATIONAL,
                intent=QueryIntent.CHITCHAT,
                complexity=ComplexityLevel.LOW,
                entities=entities,
                metadata_hints=metadata_hints,
                metadata_scope=metadata_scope,
                is_multi_question=False,
                rewrite=rewrite,
                decompose=decompose,
                rule_matches=rule_matches,
            )

        policy_ids = _POLICY_ID_PATTERN.findall(text)
        if policy_ids:
            rule_matches.append("policy_id")
            entities.extend(policy_ids)
            metadata_hints["policy_id"] = policy_ids[0]
            metadata_scope["policy_id"] = policy_ids[0]

        departments = _DEPARTMENT_PATTERN.findall(text)
        if departments:
            rule_matches.append("department")
            dept = departments[0].upper()
            metadata_hints["department"] = dept
            metadata_scope["department"] = dept

        years = _YEAR_PATTERN.findall(text)
        if years:
            rule_matches.append("year")
            metadata_hints["year"] = years[0]
            metadata_scope["year"] = years[0]

        version_match = _VERSION_PATTERN.search(text)
        if version_match:
            rule_matches.append("version")
            metadata_hints["version"] = version_match.group(0)
            metadata_scope["version"] = version_match.group(0)

        is_multi_question = len(_MULTI_QUESTION_PATTERN.findall(text)) > 1
        if is_multi_question:
            rule_matches.append("multi_question")

        if _HR_DOMAIN_PATTERN.search(text):
            rule_matches.append("hr_policy_domain")
            if len(text.split()) <= 4 and not policy_ids:
                rule_matches.append("short_keyword_query")

        intent = self._detect_intent(text, policy_ids)
        complexity = self._detect_complexity(text, is_multi_question, intent)
        query_type = self._detect_query_type(text, policy_ids, intent, complexity)
        rewrite = self._decide_rewrite(text, query_type, rule_matches)
        decompose = self._decide_decomposition(text, query_type, rule_matches)

        return self._build_analysis(
            query_type=query_type,
            intent=intent,
            complexity=complexity,
            entities=entities,
            metadata_hints=metadata_hints,
            metadata_scope=metadata_scope,
            is_multi_question=is_multi_question,
            rewrite=rewrite,
            decompose=decompose,
            rule_matches=rule_matches,
        )

    @staticmethod
    def _build_analysis(
        *,
        query_type: QueryType,
        intent: QueryIntent,
        complexity: ComplexityLevel,
        entities: list[str],
        metadata_hints: dict[str, str | int],
        metadata_scope: dict[str, str | int],
        is_multi_question: bool,
        rewrite: RewriteDecision,
        decompose: DecompositionDecision,
        rule_matches: list[str],
    ) -> QueryAnalysis:
        return QueryAnalysis(
            query_type=query_type,
            intent=intent,
            complexity=complexity,
            entities=entities,
            metadata_hints=metadata_hints,
            metadata_scope=metadata_scope,
            is_multi_question=is_multi_question,
            rewrite_decision=rewrite,
            decomposition_decision=decompose,
            needs_pre_rewrite=rewrite.should_rewrite,
            needs_decomposition=decompose.should_decompose,
            rule_matches=rule_matches,
        )

    @staticmethod
    def _detect_intent(text: str, policy_ids: list[str]) -> QueryIntent:
        if _COMPARATIVE_PATTERN.search(text):
            return QueryIntent.COMPARATIVE
        if policy_ids:
            return QueryIntent.LOOKUP
        if _SUMMARIZE_PATTERN.search(text):
            return QueryIntent.SUMMARIZATION
        if _PROCEDURAL_PATTERN.search(text):
            return QueryIntent.PROCEDURAL
        if "?" in text or len(text.split()) <= 12:
            return QueryIntent.FACTUAL
        return QueryIntent.UNKNOWN

    @staticmethod
    def _detect_query_type(
        text: str,
        policy_ids: list[str],
        intent: QueryIntent,
        complexity: ComplexityLevel,
    ) -> QueryType:
        if intent == QueryIntent.COMPARATIVE or _COMPARATIVE_PATTERN.search(text):
            return QueryType.COMPARISON
        if _BOTH_AND_PATTERN.search(text):
            return QueryType.COMPARISON
        if policy_ids:
            return QueryType.LOOKUP
        if _REWRITE_FOLLOWUP_PATTERN.search(text) or (
            len(text.split()) <= 6 and _VAGUE_PRONOUN_PATTERN.search(text)
        ):
            return QueryType.AMBIGUOUS
        if intent in (QueryIntent.SUMMARIZATION, QueryIntent.PROCEDURAL) or (
            complexity == ComplexityLevel.HIGH and intent != QueryIntent.FACTUAL
        ):
            return QueryType.SEMANTIC
        if complexity == ComplexityLevel.HIGH:
            return QueryType.MULTI_HOP
        return QueryType.FACTUAL

    @staticmethod
    def _decide_rewrite(text: str, query_type: QueryType, rule_matches: list[str]) -> RewriteDecision:
        if _REWRITE_FOLLOWUP_PATTERN.search(text):
            return RewriteDecision(should_rewrite=True, reason="context-dependent follow-up phrasing")
        if query_type == QueryType.AMBIGUOUS:
            return RewriteDecision(should_rewrite=True, reason="ambiguous query requires context resolution")
        if _VAGUE_PRONOUN_PATTERN.search(text) and len(text.split()) <= 10:
            return RewriteDecision(should_rewrite=True, reason="pronoun without clear antecedent")
        return RewriteDecision(should_rewrite=False, reason="query is sufficiently standalone")

    @staticmethod
    def _decide_decomposition(
        text: str,
        query_type: QueryType,
        rule_matches: list[str],
    ) -> DecompositionDecision:
        if _SAME_TOPIC_COMPOUND_PATTERN.search(text):
            return DecompositionDecision(
                should_decompose=False,
                reason="compound question targets a single policy topic",
            )

        if query_type == QueryType.COMPARISON and _DISTINCT_COMPARISON_PATTERN.search(text):
            return DecompositionDecision(
                should_decompose=True,
                reason="comparison across distinct entities or policies",
            )

        if _POLICY_DIFFERENCES_PATTERN.search(text):
            return DecompositionDecision(
                should_decompose=True,
                reason="side-by-side policy differences request",
            )

        if _BOTH_AND_PATTERN.search(text):
            return DecompositionDecision(
                should_decompose=True,
                reason="both/and comparison pattern detected",
            )

        if "multi_question" in rule_matches and query_type in (QueryType.COMPARISON, QueryType.MULTI_HOP):
            return DecompositionDecision(
                should_decompose=True,
                reason="multiple unrelated intents detected",
            )

        return DecompositionDecision(should_decompose=False, reason="single-intent query")

    @staticmethod
    def _detect_complexity(
        text: str,
        is_multi_question: bool,
        intent: QueryIntent,
    ) -> ComplexityLevel:
        word_count = len(text.split())
        if is_multi_question or intent in (QueryIntent.COMPARATIVE, QueryIntent.SUMMARIZATION):
            return ComplexityLevel.HIGH if word_count > 20 else ComplexityLevel.MEDIUM
        if word_count <= 8 and intent == QueryIntent.LOOKUP:
            return ComplexityLevel.LOW
        if word_count <= 12:
            return ComplexityLevel.LOW
        if word_count <= 25:
            return ComplexityLevel.MEDIUM
        return ComplexityLevel.HIGH
