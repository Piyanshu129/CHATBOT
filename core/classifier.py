"""Query classification module for routing queries to appropriate retrieval strategies."""

import logging
import re
from enum import Enum
from typing import Optional, Any
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class QueryCategory(Enum):
    """Categories for query classification."""

    CHAT = "chat"  # Small talk, greetings, general talking
    PERSONAL_MEMORY = "personal_memory"  # Asking about previous conversations
    TASK = "task"  # Instructions like generate, write, summarize, create
    TECHNICAL_QA = "technical_qa"  # Coding, ML, APIs, software
    MATH_OR_CODE = "math_or_code"  # Equations, debugging, coding help
    SEARCH_RELEVANT = "search_relevant"  # Factual knowledge requiring retrieval
    NO_RETRIEVAL = "no_retrieval"  # Questions answerable without context


@dataclass
class ClassificationResult:
    """Result of query classification."""

    category: QueryCategory
    confidence: float
    needs_retrieval: bool
    retrieval_strategy: str


class QueryClassifier:
    """Classifies user queries into categories for intelligent routing."""

    # Pattern definitions for rule-based classification
    CHAT_PATTERNS = [
        r"\b(hello|hi|hey|greetings|good morning|good evening)\b",
        r"\bhow are you\b",
        r"\bnice to meet you\b",
        r"\bthanks?|thank you\b",
        r"\bbye|goodbye|see you\b",
    ]

    PERSONAL_MEMORY_PATTERNS = [
        r"\b(what|tell me|remind me).*(did (i|we)|said|discussed|talked about|mentioned)\b",
        r"\b(earlier|before|previous(ly)?|last time)\b",
        r"\bour (previous |last )?conversation\b",
        r"\bwhat (did (i|we)|have (i|we))\b",
    ]

    TASK_PATTERNS = [
        r"\b(write|create|generate|make|build|design|develop|compose)\b",
        r"\b(summarize|explain|describe|outline|list)\b",
        r"\b(translate|convert|transform|rewrite)\b",
        r"\bhelp me (with|to)\b",
    ]

    TECHNICAL_QA_PATTERNS = [
        r"\b(code|coding|program(ming)?|script|function|class|method)\b",
        r"\b(api|sdk|library|framework|package|module)\b",
        r"\b(machine learning|deep learning|neural network|model|algorithm)\b",
        r"\b(database|sql|query|table|schema)\b",
        r"\b(python|javascript|java|c\+\+|rust|go)\b",
        r"\b(debug|error|exception|bug|issue)\b",
    ]

    MATH_OR_CODE_PATTERNS = [
        r"\b(solve|calculate|compute|evaluate)\b",
        r"\b(equation|formula|expression|derivative|integral)\b",
        r"\b(debug|fix|refactor|optimize) (this |the )?code\b",
        r"[\+\-\*/=]\s*[\d\w]+",  # Math expressions
        r"(def |class |function |import |return )",  # Code snippets
    ]

    SEARCH_RELEVANT_PATTERNS = [
        r"\b(what is|what are|who is|who are|when did|where is|how does)\b",
        r"\b(define|definition|meaning of)\b",
        r"\b(history of|background on)\b",
        r"\b(fact|information|detail)s? about\b",
    ]

    NO_RETRIEVAL_PATTERNS = [
        r"\b(simple |basic )?(question|answer)\b",
        r"\b(yes|no) or (yes|no)\b",
        r"\b(true|false)\b",
        r"^\w+\?$",  # Single word questions
    ]

    def _llm_classify(self, query: str) -> ClassificationResult:
        """
        LLM-based classification for more accurate intent detection.
        This is used when classifier_type = 'llm'.
        """
        try:
            # Build system + user prompt
            sys_prompt = (
                "Classify the user's query into exactly one of these categories:\n"
                "chat, personal_memory, task, technical_qa, math_or_code, search_relevant, no_retrieval.\n"
                "Return only the category name. No explanation."
            )

            messages = [SystemMessage(content=sys_prompt), HumanMessage(content=query)]

            # Call LLM
            result = self.llm.invoke(messages)  # you will attach llm externally
            predicted = result.content.strip().lower()

            # Validate result
            if predicted not in [c.value for c in QueryCategory]:
                predicted = "search_relevant"  # fallback

            category = QueryCategory(predicted)

            # Same routing logic as rule-based
            needs_retrieval, retrieval_strategy = self._get_retrieval_info(category)

            return ClassificationResult(
                category=category,
                confidence=0.9,
                needs_retrieval=needs_retrieval,
                retrieval_strategy=retrieval_strategy,
            )

        except Exception as e:
            logger.error(f"LLM classification failed, falling back to rule-based: {e}")
            return self._rule_based_classify(query)

    def __init__(self, classifier_type: str = "llm", llm: Optional[Any] = None):
        """
        Initialize query classifier.

        Args:
            classifier_type: Type of classifier ("rule_based" or "llm")
            llm: Language model instance (required for llm classification)
        """
        self.classifier_type = classifier_type
        self.llm = llm
        logger.info(f"Initialized QueryClassifier with type: {self.classifier_type}")

    def classify(self, query: str) -> ClassificationResult:
        """
        Classify a user query into one of the predefined categories.
        """
        if self.classifier_type == "rule_based":
            return self._rule_based_classify(query)

        elif self.classifier_type == "llm":
            return self._llm_classify(query)

        else:
            logger.warning(
                f"Unknown classifier_type '{self.classifier_type}', falling back to rule-based."
            )
            return self._llm_classify(query)

    def _rule_based_classify(self, query: str) -> ClassificationResult:
        """
        Rule-based classification using pattern matching.

        Args:
            query: User's input query

        Returns:
            ClassificationResult
        """
        query_lower = query.lower()

        # Rule: If query contains NO nouns (heuristic: all words are stop words), skip retrieval
        # This is a basic filter to prevent retrieval for queries like "is it", "did you", etc.
        stop_words = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "from",
            "up",
            "down",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
            "d",
            "ll",
            "m",
            "o",
            "re",
            "ve",
            "y",
            "ain",
            "aren",
            "couldn",
            "didn",
            "doesn",
            "hadn",
            "hasn",
            "haven",
            "isn",
            "ma",
            "mightn",
            "mustn",
            "needn",
            "shan",
            "shouldn",
            "wasn",
            "weren",
            "won",
            "wouldn",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "about",
            "against",
            "between",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "to",
            "from",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
        }

        words = query_lower.split()
        content_words = [w for w in words if w not in stop_words]

        # If no content words (likely no nouns), default to NO_RETRIEVAL
        if not content_words:
            logger.info("Query contains no content words (nouns), skipping retrieval.")
            return ClassificationResult(
                category=QueryCategory.NO_RETRIEVAL,
                confidence=1.0,
                needs_retrieval=False,
                retrieval_strategy="none",
            )

        # Rule: If query contains NO question words and is short, might be chat
        question_words = {
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "which",
            "whose",
            "whom",
        }
        has_question_word = any(w in words for w in question_words)

        # Check patterns in priority order
        scores = {
            QueryCategory.CHAT: self._pattern_score(query_lower, self.CHAT_PATTERNS),
            QueryCategory.PERSONAL_MEMORY: self._pattern_score(
                query_lower, self.PERSONAL_MEMORY_PATTERNS
            ),
            QueryCategory.MATH_OR_CODE: self._pattern_score(
                query_lower, self.MATH_OR_CODE_PATTERNS
            ),
            QueryCategory.TECHNICAL_QA: self._pattern_score(
                query_lower, self.TECHNICAL_QA_PATTERNS
            ),
            QueryCategory.TASK: self._pattern_score(query_lower, self.TASK_PATTERNS),
            QueryCategory.SEARCH_RELEVANT: self._pattern_score(
                query_lower, self.SEARCH_RELEVANT_PATTERNS
            ),
            QueryCategory.NO_RETRIEVAL: self._pattern_score(
                query_lower, self.NO_RETRIEVAL_PATTERNS
            ),
        }

        # Get category with highest score
        best_category = max(scores.items(), key=lambda x: x[1])
        category, confidence = best_category

        # If no strong match
        if confidence < 0.3:
            # If it has no question words and wasn't classified as anything else strongly,
            # it's likely not a search query.
            if not has_question_word:
                category = (
                    QueryCategory.CHAT
                )  # Default to CHAT instead of SEARCH if no question word
                confidence = 0.5
            else:
                category = QueryCategory.SEARCH_RELEVANT
                confidence = 0.5

        # Determine retrieval needs
        needs_retrieval, retrieval_strategy = self._get_retrieval_info(category)

        logger.debug(
            f"Classified query as {category.value} with confidence {confidence:.2f}"
        )

        return ClassificationResult(
            category=category,
            confidence=confidence,
            needs_retrieval=needs_retrieval,
            retrieval_strategy=retrieval_strategy,
        )

    def _pattern_score(self, query: str, patterns: list) -> float:
        """
        Calculate score based on pattern matches.

        Args:
            query: Query text (lowercased)
            patterns: List of regex patterns

        Returns:
            Score between 0 and 1
        """
        matches = 0
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matches += 1

        # Normalize by number of patterns
        return min(matches / max(len(patterns) / 2, 1), 1.0)

    def _get_retrieval_info(self, category: QueryCategory) -> tuple[bool, str]:
        """
        Determine retrieval needs and strategy based on category.

        Args:
            category: Query category

        Returns:
            Tuple of (needs_retrieval, retrieval_strategy)
        """
        retrieval_config = {
            QueryCategory.CHAT: (False, "none"),
            QueryCategory.PERSONAL_MEMORY: (True, "short_term_only"),
            QueryCategory.TASK: (True, "hybrid_recent"),
            QueryCategory.TECHNICAL_QA: (True, "hybrid_full"),
            QueryCategory.MATH_OR_CODE: (True, "hybrid_full"),
            QueryCategory.SEARCH_RELEVANT: (True, "hybrid_full"),
            QueryCategory.NO_RETRIEVAL: (False, "none"),
        }

        return retrieval_config.get(category, (True, "hybrid_full"))

    def should_use_retrieval(self, category: QueryCategory) -> bool:
        """
        Check if retrieval is needed for a category.

        Args:
            category: Query category

        Returns:
            True if retrieval should be used
        """
        needs_retrieval, _ = self._get_retrieval_info(category)
        return needs_retrieval

    def get_retrieval_strategy(self, category: QueryCategory) -> str:
        """
        Get retrieval strategy for a category.

        Args:
            category: Query category

        Returns:
            Retrieval strategy name
        """
        _, strategy = self._get_retrieval_info(category)
        return strategy
