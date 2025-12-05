"""Metadata filtering module for context-aware document filtering."""

import logging
import time
from typing import List, Optional
from langchain_core.documents import Document

from ..config import BotConfig
from .classifier import QueryCategory

logger = logging.getLogger(__name__)


class MetadataFilter:
    """Filters documents based on metadata criteria."""

    def __init__(self, config: BotConfig):
        """
        Initialize metadata filter.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.recency_seconds = config.recency_days * 24 * 60 * 60
        self.min_score = config.min_relevance_score

        logger.info("Initialized MetadataFilter")

    def filter(
        self,
        documents: List[Document],
        category: Optional[QueryCategory] = None,
        topic: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> List[Document]:
        """
        Apply filters to documents.

        Args:
            documents: List of documents to filter
            category: Query category for filtering
            topic: Topic filter
            min_score: Minimum relevance score (overrides config)

        Returns:
            Filtered list of documents
        """
        if not self.config.enable_metadata_filtering:
            logger.debug("Metadata filtering disabled")
            return documents

        if not documents:
            return []

        filtered = documents

        # Apply category filter
        if category:
            filtered = self.filter_by_category(filtered, category)

        # Apply recency filter (for recent memory needs)
        if category in [QueryCategory.PERSONAL_MEMORY, QueryCategory.TASK]:
            filtered = self.filter_by_recency(filtered, self.config.recency_days)

        # Apply topic filter
        if topic:
            filtered = self.filter_by_topic(filtered, topic)

        # Apply score threshold
        score_threshold = min_score or self.min_score
        filtered = self.filter_by_score(filtered, score_threshold)

        logger.debug(f"Filtered {len(documents)} documents to {len(filtered)}")
        return filtered

    def filter_by_category(
        self, documents: List[Document], category: QueryCategory
    ) -> List[Document]:
        """
        Filter documents by category.

        Args:
            documents: List of documents
            category: Query category

        Returns:
            Filtered documents
        """
        logger.info(
            f"DEBUG: filter_by_category with category={category.value}, {len(documents)} docs"
        )
        filtered = []
        for i, doc in enumerate(documents):
            doc_category = doc.metadata.get("category") if doc.metadata else None
            logger.info(f"DEBUG: Doc {i} has category='{doc_category}'")

            # If no category metadata, include document (backward compatibility)
            if doc_category is None:
                logger.info(f"DEBUG: Doc {i}: No category, including")
                filtered.append(doc)
            # "interaction" is legacy default - matches everything (backward compatibility)
            elif doc_category == "interaction":
                logger.info(
                    f"DEBUG: Doc {i}: Category is 'interaction' (legacy), including"
                )
                filtered.append(doc)
            # Match category
            elif doc_category == category.value:
                logger.info(
                    f"DEBUG: Doc {i}: Category matches {category.value}, including"
                )
                filtered.append(doc)
            # Also include if categories are compatible
            elif self._categories_compatible(category, doc_category):
                logger.info(
                    f"DEBUG: Doc {i}: Category compatible with {category.value}, including"
                )
                filtered.append(doc)
            else:
                logger.info(
                    f"DEBUG: Doc {i}: Category '{doc_category}' doesn't match {category.value}, FILTERED OUT"
                )

        logger.info(
            f"DEBUG: filter_by_category: {len(documents)} → {len(filtered)} docs"
        )
        return filtered

    def filter_by_recency(self, documents: List[Document], days: int) -> List[Document]:
        """
        Filter documents by recency.

        Args:
            documents: List of documents
            days: Number of days (documents older than this are filtered out)

        Returns:
            Recent documents
        """
        current_time = time.time()
        cutoff_time = current_time - (days * 24 * 60 * 60)

        filtered = []
        for doc in documents:
            timestamp = doc.metadata.get("timestamp") if doc.metadata else None

            # If no timestamp, include document (backward compatibility)
            if timestamp is None:
                filtered.append(doc)
            # Check if recent enough
            elif timestamp >= cutoff_time:
                filtered.append(doc)

        return filtered

    def filter_by_topic(self, documents: List[Document], topic: str) -> List[Document]:
        """
        Filter documents by topic.

        Args:
            documents: List of documents
            topic: Topic string

        Returns:
            Documents matching topic
        """
        topic_lower = topic.lower()

        filtered = []
        for doc in documents:
            doc_topic = doc.metadata.get("topic") if doc.metadata else None

            # If no topic metadata, include document
            if doc_topic is None:
                filtered.append(doc)
            # Check if topic matches
            elif doc_topic.lower() == topic_lower:
                filtered.append(doc)
            # Also check if topic is in document content
            elif topic_lower in doc.page_content.lower():
                filtered.append(doc)

        return filtered

    def filter_by_score(
        self, documents: List[Document], min_score: float
    ) -> List[Document]:
        """
        Filter documents by relevance score.

        Args:
            documents: List of documents
            min_score: Minimum score threshold

        Returns:
            Documents with score >= min_score
        """
        logger.info(
            f"DEBUG: filter_by_score called with min_score={min_score}, {len(documents)} docs"
        )
        filtered = []
        for i, doc in enumerate(documents):
            # Check for rerank score first (most accurate)
            score = doc.metadata.get("rerank_score") if doc.metadata else None

            # Fall back to retrieval score if available
            if score is None and doc.metadata:
                score = doc.metadata.get("score")

            logger.info(f"DEBUG: Doc {i}: score={score}, metadata={doc.metadata}")

            # If no score, include document (backward compatibility)
            if score is None:
                logger.info(f"DEBUG: Doc {i}: No score, including")
                filtered.append(doc)
            # Check if score meets threshold
            elif score >= min_score:
                logger.info(f"DEBUG: Doc {i}: Score {score} >= {min_score}, including")
                filtered.append(doc)
            else:
                logger.info(
                    f"DEBUG: Doc {i}: Score {score} < {min_score}, FILTERED OUT"
                )

        return filtered

    def _categories_compatible(self, query_cat: QueryCategory, doc_cat: str) -> bool:
        """
        Check if query category and document category are compatible.

        Args:
            query_cat: Query category
            doc_cat: Document category string

        Returns:
            True if compatible
        """
        # Define compatible category groups
        compatible_groups = [
            {QueryCategory.TECHNICAL_QA.value, QueryCategory.MATH_OR_CODE.value},
            {QueryCategory.TASK.value, QueryCategory.SEARCH_RELEVANT.value},
        ]

        query_value = query_cat.value

        for group in compatible_groups:
            if query_value in group and doc_cat in group:
                return True

        return False
