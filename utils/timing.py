"""Timing utilities for the chatbot."""

import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimingStats:
    """Class to hold timing statistics."""

    model_loading: float = 0.0
    embedding_generation: float = 0.0
    short_term_memory: float = 0.0
    long_term_memory: float = 0.0
    combined_memory: float = 0.0
    ttft: float = 0.0  # Time to First Token

    # Production RAG enhancements
    query_classification: float = 0.0
    query_expansion: float = 0.0
    bm25_retrieval: float = 0.0
    hybrid_fusion: float = 0.0
    reranking: float = 0.0
    metadata_filtering: float = 0.0
    context_assembly: float = 0.0


class TimingManager:
    """Singleton to manage timing statistics."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimingManager, cls).__new__(cls)
            cls._instance.stats = TimingStats()
        return cls._instance

    def reset(self):
        """Reset all statistics."""
        self.stats = TimingStats()

    def get_stats(self) -> Dict[str, float]:
        """Get current statistics as a dictionary."""
        return {
            "model_loading_time": self.stats.model_loading,
            "embedding_generation_time": self.stats.embedding_generation,
            "short_term_memory_query_time": self.stats.short_term_memory,
            "long_term_memory_query_time": self.stats.long_term_memory,
            "combined_memory_query_time": self.stats.combined_memory,
            "time_to_first_token": self.stats.ttft,
            # Production RAG enhancements
            "query_classification_time": self.stats.query_classification,
            "query_expansion_time": self.stats.query_expansion,
            "bm25_retrieval_time": self.stats.bm25_retrieval,
            "hybrid_fusion_time": self.stats.hybrid_fusion,
            "reranking_time": self.stats.reranking,
            "metadata_filtering_time": self.stats.metadata_filtering,
            "context_assembly_time": self.stats.context_assembly,
        }


class TimingContext:
    """Context manager for measuring execution time."""

    def __init__(self, metric_name: str, manager: Optional[TimingManager] = None):
        self.metric_name = metric_name
        self.manager = manager or TimingManager()
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time

        if hasattr(self.manager.stats, self.metric_name):
            setattr(self.manager.stats, self.metric_name, duration)
            logger.debug(f"Timing - {self.metric_name}: {duration:.4f}s")
        else:
            logger.warning(f"Unknown timing metric: {self.metric_name}")
