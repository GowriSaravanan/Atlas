"""Rule-based query decomposition policy."""

from __future__ import annotations

import re

from adaptive_rag.domain.models.decomposition import DecompositionResult, SubQuery, SubQuerySource
from adaptive_rag.domain.models.query import QueryType, ResolvedQueryAnalysis
from adaptive_rag.domain.policies.query_analyzer import QueryAnalyzer
from adaptive_rag.domain.ports.query_decomposer import QueryDecomposerPort

_COMPARE_AND_PATTERN = re.compile(
    r"\bcompare\s+(?P<left>.+?)\s+(?:and|with)\s+(?P<right>.+?)(?:\s+benefits?|\s+policies?)?\.?\s*$",
    re.IGNORECASE,
)
_COMPARE_VERSUS_PATTERN = re.compile(
    r"\bcompare\s+(?P<left>.+?)\s+(?:versus|vs\.?)\s+(?P<right>.+?)(?:\s+benefits?|\s+policies?|\s+allowances)?\.?\s*$",
    re.IGNORECASE,
)
_DIFFERENCE_PATTERN = re.compile(
    r"\bdifference between\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\s+benefits?|\s+policies?)?\.?\s*$",
    re.IGNORECASE,
)
_VS_PATTERN = re.compile(
    r"^(?P<left>.+?)\s+(?:vs\.?|versus)\s+(?P<right>.+?)(?:\s+benefits?|\s+policies?)?\.?\s*$",
    re.IGNORECASE,
)
_VERSUS_PREFIX_PATTERN = re.compile(
    r"^\s*(?:versus|vs\.?)\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\s+benefits?)?\.?\s*$",
    re.IGNORECASE,
)
_BOTH_AND_PATTERN = re.compile(
    r"^\s*both\s+(?P<left>.+?)\s+and\s+(?P<right>.+?)(?:\s+benefits?)?\.?\s*$",
    re.IGNORECASE,
)
_AND_DIFFERENCES_PATTERN = re.compile(
    r"^(?P<left>.+?)\s+and\s+(?P<right>.+?)\s+differences?\.?\s*$",
    re.IGNORECASE,
)
_POLICY_ID_PATTERN = re.compile(r"\b([A-Z]{2,10}-\d{2,6})\b")


class RuleBasedQueryDecomposer(QueryDecomposerPort):
    """Deterministic query decomposer for comparisons and pass-through queries."""

    def __init__(self, query_analyzer: QueryAnalyzer | None = None) -> None:
        self._analyzer = query_analyzer or QueryAnalyzer()

    def decompose(
        self,
        query: str,
        analysis: ResolvedQueryAnalysis,
    ) -> DecompositionResult:
        """Split comparison queries or return a single pass-through subquery."""
        text = query.strip()
        if not analysis.decomposition_decision.should_decompose:
            return self._pass_through(text, analysis, reason=analysis.decomposition_decision.reason)

        pairs = self._extract_comparison_pair(text)
        if pairs is None:
            return self._pass_through(
                text,
                analysis,
                reason="decomposition requested but no safe rule-based split applied",
            )

        left, right = pairs
        left = RuleBasedQueryDecomposer._clean_comparison_entity(left)
        right = RuleBasedQueryDecomposer._clean_comparison_entity(right)
        subqueries = [
            self._build_subquery("A", left, text, SubQuerySource.COMPARISON),
            self._build_subquery("B", right, text, SubQuerySource.COMPARISON),
        ]
        return DecompositionResult(
            original_query=text,
            was_decomposed=True,
            reason=analysis.decomposition_decision.reason,
            subqueries=subqueries,
        )

    def _pass_through(
        self,
        query: str,
        analysis: ResolvedQueryAnalysis,
        *,
        reason: str,
    ) -> DecompositionResult:
        return DecompositionResult(
            original_query=query,
            was_decomposed=False,
            reason=reason or "single-intent query",
            subqueries=[
                SubQuery(
                    id="0",
                    query=query,
                    entity=None,
                    source=SubQuerySource.PASS_THROUGH,
                    query_type=analysis.query_type,
                    parent_query=query,
                )
            ],
        )

    def _build_subquery(
        self,
        subquery_id: str,
        entity: str,
        parent_query: str,
        source: SubQuerySource,
    ) -> SubQuery:
        normalized = self._normalize_entity_query(entity.strip(" ."))
        sub_analysis = self._analyzer.analyze(normalized)
        return SubQuery(
            id=subquery_id,
            query=normalized,
            entity=entity.strip(" ."),
            source=source,
            query_type=sub_analysis.query_type,
            parent_query=parent_query,
        )

    @staticmethod
    def _clean_comparison_entity(entity: str) -> str:
        text = entity.strip(" .")
        policy_match = _POLICY_ID_PATTERN.search(text)
        if policy_match:
            return policy_match.group(1)
        return re.sub(
            r"\s+(leave\s+)?(allowances|benefits?|policies?|details?)$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    @staticmethod
    def _normalize_entity_query(entity: str) -> str:
        entity = entity.strip(" .")
        if _POLICY_ID_PATTERN.search(entity):
            return f"What is {entity}?"
        entity_lower = entity.lower()
        if entity_lower.endswith(" policy") or entity_lower.endswith(" benefits"):
            return f"What is the {entity}?"
        if "policy" in entity_lower or "leave" in entity_lower or "benefit" in entity_lower:
            return f"What is the {entity} policy?"
        return f"What is the {entity} policy?"

    @staticmethod
    def _extract_comparison_pair(text: str) -> tuple[str, str] | None:
        for pattern in (
            _COMPARE_AND_PATTERN,
            _COMPARE_VERSUS_PATTERN,
            _DIFFERENCE_PATTERN,
            _VERSUS_PREFIX_PATTERN,
            _BOTH_AND_PATTERN,
            _AND_DIFFERENCES_PATTERN,
            _VS_PATTERN,
        ):
            match = pattern.search(text)
            if match:
                left = match.group("left").strip(" .")
                right = match.group("right").strip(" .")
                if left and right:
                    return left, right
        return None
