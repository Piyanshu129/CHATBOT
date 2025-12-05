"""Retrieval guardrails to prevent unnecessary RAG execution."""

import logging
import re
from typing import List, Set

from ..config import BotConfig
from .classifier import QueryCategory

logger = logging.getLogger(__name__)


class RetrievalGuard:
    """
    Guardrails to determine if retrieval should be performed.
    Prevents RAG for simple greetings, thanks, and non-factual queries.
    """

    # Words that indicate a need for retrieval
    RETRIEVAL_KEYWORDS = {
        "buy",
        "recommend",
        "suggest",
        "find",
        "search",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "explain",
        "describe",
        "list",
        "show",
        "tell",
        "price",
        "cost",
        "budget",
        "specs",
        "features",
        "compare",
        "difference",
        "best",
        "top",
        "cheapest",
        "expensive",
        "review",
        "opinion",
    }

    # Words that indicate NO need for retrieval (if they are the main content)
    SKIP_KEYWORDS = {
        "hi",
        "hello",
        "hey",
        "greetings",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "thx",
        "cool",
        "ok",
        "okay",
        "bye",
        "goodbye",
        "cya",
        "exit",
        "quit",
        "help",
    }

    def __init__(self, config: BotConfig):
        self.config = config
        logger.info("Initialized RetrievalGuard")

    def should_retrieve(self, query: str, category: QueryCategory) -> bool:
        """
        Determine if retrieval should be performed for the query.

        Logic:
        If ANY of the following are true, ALLOW retrieval:
        1. Requires facts (based on category or keywords)
        2. Depends on history (based on keywords)
        3. Contains specific retrieval keywords ("buy", "recommend", etc.)
        4. Contains code keywords

        Otherwise, SKIP retrieval.
        """
        query_lower = query.lower().strip()
        words = query_lower.split()

        # Rule 1: Skip if query is very short and contains skip keywords
        if len(words) <= 3 and any(w in self.SKIP_KEYWORDS for w in words):
            logger.info(f"Skipping retrieval: Short query with skip keyword '{query}'")
            return False

        # Check 1: Does query require facts?
        # We use the category for this, as the classifier is trained/designed to detect this.
        requires_facts = category in [
            QueryCategory.TECHNICAL_QA,
            QueryCategory.SEARCH_RELEVANT,
            QueryCategory.MATH_OR_CODE,
            QueryCategory.TASK,
        ]
        if requires_facts:
            logger.info(f"Allowing retrieval: Category {category.value} requires facts")
            return True

        # Check 2: Does query depend on history?
        depends_on_history = self._check_dependency_on_history(query_lower)
        if depends_on_history:
            logger.info("Allowing retrieval: Query depends on history")
            return True

        # Check 3: Does query contain "buy", "recommend", "tell me about"?
        # We use our existing RETRIEVAL_KEYWORDS plus "tell me about"
        has_keywords = (
            any(w in self.RETRIEVAL_KEYWORDS for w in words)
            or "tell me about" in query_lower
        )
        if has_keywords:
            logger.info("Allowing retrieval: Contains retrieval keywords")
            return True

        # Check 4: Does query contain code keywords?
        has_code = self._check_code_keywords(query_lower)
        if has_code:
            logger.info("Allowing retrieval: Contains code keywords")
            return True

        # If none of the above, SKIP
        logger.info("Skipping retrieval: No retrieval triggers found (Guardrails)")
        return False

    def _check_dependency_on_history(self, query: str) -> bool:
        """Check if query implies dependency on previous conversation."""
        history_keywords = {
            "remember",
            "said",
            "told",
            "last",
            "previous",
            "earlier",
            "ago",
            "before",
            "mentioned",
            "discussed",
            "we talked",
            "you said",
        }
        return any(w in query for w in history_keywords)

    def _check_code_keywords(self, query: str) -> bool:
        """Check if query contains programming/technical keywords."""
        code_keywords = {
            "code",
            "python",
            "java",
            "script",
            "function",
            "class",
            "error",
            "bug",
            "debug",
            "api",
            "library",
            "framework",
            "install",
            "run",
            "compile",
            "build",
            "deploy",
            "git",
            "database",
            "sql",
            "json",
            "xml",
            "html",
            "css",
            "variable",
            "loop",
            "import",
        }
        words = set(query.split())
        return not words.isdisjoint(code_keywords)
