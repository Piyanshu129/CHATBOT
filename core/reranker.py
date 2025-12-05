"""Cross-encoder reranking module for improving retrieval precision."""

import logging
from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from ..config import BotConfig

logger = logging.getLogger(__name__)


class Reranker:
    """Reranks retrieved documents using a cross-encoder model."""

    def __init__(self, config: BotConfig):
        """
        Initialize reranker.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.top_k = config.rerank_top_k
        self.model = None

        if config.enable_reranking:
            self._load_model()

        logger.info(f"Initialized Reranker with model: {config.reranker_model}")

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            logger.info(f"Loading cross-encoder model: {self.config.reranker_model}")
            self.model = CrossEncoder(self.config.reranker_model)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            logger.warning("Reranking will be disabled")
            self.model = None

    def rerank(
        self, query: str, documents: List[Document], top_k: int = None
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Original query
            documents: List of documents to rerank
            top_k: Number of top documents to return (default: use config value)

        Returns:
            Reranked list of documents (best first)
        """
        if not self.config.enable_reranking or not self.model:
            logger.debug("Reranking disabled, returning original documents")
            return documents[: top_k or self.top_k]

        if not documents:
            logger.warning("No documents to rerank")
            return []

        top_k = top_k or self.top_k

        # If fewer docs than requested, return all
        if len(documents) <= top_k:
            logger.debug(
                f"Document count ({len(documents)}) <= top_k ({top_k}), skipping reranking"
            )
            return documents

        try:
            # Prepare query-document pairs
            pairs = [[query, doc.page_content] for doc in documents]

            # Get scores from cross-encoder
            logger.debug(f"Reranking {len(documents)} documents")
            scores = self.model.predict(pairs)

            # Combine documents with scores
            doc_score_pairs = list(zip(documents, scores))

            # Sort by score (descending)
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

            # Extract top-k documents
            reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]

            # Store scores in metadata for debugging
            for i, (doc, score) in enumerate(doc_score_pairs[:top_k]):
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["rerank_score"] = float(score)
                doc.metadata["rerank_position"] = i

            logger.debug(f"Reranked to top {top_k} documents")
            return reranked_docs

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            logger.warning("Falling back to original document order")
            return documents[:top_k]

    def get_scores(self, query: str, documents: List[Document]) -> List[float]:
        """
        Get relevance scores for documents without reranking.

        Args:
            query: Original query
            documents: List of documents

        Returns:
            List of relevance scores
        """
        if not self.model or not documents:
            return [0.0] * len(documents)

        try:
            pairs = [[query, doc.page_content] for doc in documents]
            scores = self.model.predict(pairs)
            return scores.tolist()
        except Exception as e:
            logger.error(f"Score computation failed: {e}")
            return [0.0] * len(documents)
