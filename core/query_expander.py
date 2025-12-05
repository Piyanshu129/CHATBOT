"""Multi-query expansion module for improving retrieval recall."""

import logging
from typing import List
from functools import lru_cache

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import BotConfig

logger = logging.getLogger(__name__)


class QueryExpander:
    """Expands queries into multiple variants to improve retrieval recall."""

    EXPANSION_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a query expansion expert. Given a user question, generate {num_variants} alternative phrasings that preserve the original intent but use different wording.

Rules:
1. Each variant should be semantically similar but lexically different
2. Use synonyms, different sentence structures, and varied terminology
3. Keep variants concise and focused
4. Each variant on a new line
5. Do NOT number the variants
6. Do NOT include the original query

<|eot_id|><|start_header_id|>user<|end_header_id|>
Original question: {question}

Generate {num_variants} alternative phrasings:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    # Template-based fallback patterns for common query types
    TEMPLATE_EXPANSIONS = {
        "what_is": [
            "Define {topic}",
            "Explain {topic}",
            "{topic} meaning",
            "Tell me about {topic}",
        ],
        "how_to": [
            "Guide for {action}",
            "Steps to {action}",
            "Tutorial on {action}",
            "{action} instructions",
        ],
        "why": [
            "Reason for {topic}",
            "Cause of {topic}",
            "{topic} explanation",
            "Why does {topic} happen",
        ],
    }

    def __init__(self, config: BotConfig, llm: BaseLLM = None):
        """
        Initialize query expander.

        Args:
            config: Bot configuration
            llm: Language model for expansion (optional)
        """
        self.config = config
        self.llm = llm
        self.num_variants = config.num_query_variants
        self.cache_size = config.expansion_cache_size

        # Build LLM chain if LLM provided
        if llm:
            self.prompt = ChatPromptTemplate.from_template(
                self.EXPANSION_PROMPT_TEMPLATE
            )
            self.chain = self.prompt | llm | StrOutputParser()
        else:
            self.chain = None

        logger.info(f"Initialized QueryExpander with {self.num_variants} variants")

    def expand(self, query: str) -> List[str]:
        """
        Expand query into multiple variants.

        Args:
            query: Original user query

        Returns:
            List of query variants (including original)
        """
        if not self.config.enable_query_expansion:
            logger.debug("Query expansion disabled, returning original query only")
            return [query]

        # Try cached expansion first
        try:
            variants = self._expand_cached(query)
        except Exception as e:
            logger.warning(f"Expansion failed: {e}, using original query only")
            variants = [query]

        logger.debug(f"Expanded query into {len(variants)} variants")
        return variants

    @lru_cache(maxsize=128)
    def _expand_cached(self, query: str) -> List[str]:
        """
        Cached query expansion.

        Args:
            query: Original query

        Returns:
            List of query variants
        """
        variants = [query]  # Always include original

        if self.chain:
            # LLM-based expansion
            try:
                expanded = self._expand_with_llm(query)
                variants.extend(expanded)
            except Exception as e:
                logger.warning(f"LLM expansion failed: {e}, trying template-based")
                expanded = self._expand_with_templates(query)
                variants.extend(expanded)
        else:
            # Template-based expansion fallback
            expanded = self._expand_with_templates(query)
            variants.extend(expanded)

        # Deduplicate and limit
        variants = list(
            dict.fromkeys(variants)
        )  # Remove duplicates while preserving order
        return variants[: self.num_variants + 1]  # +1 for original

    def _expand_with_llm(self, query: str) -> List[str]:
        """
        Expand query using LLM.

        Args:
            query: Original query

        Returns:
            List of expanded variants
        """
        logger.debug("Expanding query with LLM")

        response = self.chain.invoke(
            {"question": query, "num_variants": self.num_variants}
        )

        # Parse response - each variant on a new line
        variants = [
            line.strip()
            for line in response.strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]

        # Clean up any numbering
        variants = [self._clean_variant(v) for v in variants]

        return variants[: self.num_variants]

    def _expand_with_templates(self, query: str) -> List[str]:
        """
        Expand query using template-based patterns.

        Args:
            query: Original query

        Returns:
            List of expanded variants
        """
        logger.debug("Expanding query with templates")

        query_lower = query.lower()
        variants = []

        # Detect query type and apply templates
        if query_lower.startswith("what is ") or query_lower.startswith("what are "):
            topic = (
                query[8:].strip()
                if query_lower.startswith("what is ")
                else query[9:].strip()
            )
            variants = [
                t.format(topic=topic)
                for t in self.TEMPLATE_EXPANSIONS["what_is"][: self.num_variants]
            ]

        elif query_lower.startswith("how to ") or query_lower.startswith("how do i "):
            action = (
                query[7:].strip()
                if query_lower.startswith("how to ")
                else query[10:].strip()
            )
            variants = [
                t.format(action=action)
                for t in self.TEMPLATE_EXPANSIONS["how_to"][: self.num_variants]
            ]

        elif query_lower.startswith("why "):
            topic = query[4:].strip()
            variants = [
                t.format(topic=topic)
                for t in self.TEMPLATE_EXPANSIONS["why"][: self.num_variants]
            ]

        else:
            # Generic expansion - simple paraphrasing
            variants = [
                f"Tell me about {query}",
                f"Explain {query}",
                f"Information on {query}",
                f"Details about {query}",
            ][: self.num_variants]

        return variants

    def _clean_variant(self, variant: str) -> str:
        """
        Clean variant text by removing numbering and extra whitespace.

        Args:
            variant: Variant text

        Returns:
            Cleaned variant
        """
        # Remove leading numbers like "1. ", "2) ", etc.
        variant = variant.lstrip("0123456789.-)> ")
        return variant.strip()

    def clear_cache(self) -> None:
        """Clear the expansion cache."""
        self._expand_cached.cache_clear()
        logger.info("Cleared query expansion cache")
