"""SPLADE handler for sparse retrieval."""

import logging
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Dict, List, Tuple

from ..config import BotConfig

logger = logging.getLogger(__name__)


class SpladeHandler:
    """Handles SPLADE model loading and inference for sparse retrieval."""

    def __init__(self, config: BotConfig):
        """
        Initialize SPLADE handler.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.model_id = config.splade_model_id
        self.tokenizer = None
        self.model = None

        if config.enable_splade:
            self._load_model()

    def _load_model(self) -> None:
        """Load the SPLADE model and tokenizer."""
        try:
            logger.info(f"Loading SPLADE model: {self.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_id)
            self.model.eval()
            logger.info("SPLADE model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SPLADE model: {e}")
            logger.warning("SPLADE retrieval will be disabled")
            self.model = None

    def encode(self, text: str) -> Dict[int, float]:
        """
        Generate sparse embedding for text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping token IDs to weights
        """
        if not self.model or not text:
            return {}

        try:
            # Tokenize
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )

            with torch.no_grad():
                # Generate logits
                logits = self.model(**inputs).logits

            # SPLADE max pooling (log(1 + relu(logits)))
            # We only care about the masked tokens, but for query/doc we usually use the whole sequence
            # Standard SPLADE: max over sequence length
            values, _ = torch.max(torch.log(1 + torch.relu(logits)), dim=1)

            # Get non-zero values
            # values is [1, vocab_size]
            vector = values[0]

            # Extract non-zero indices and values
            indices = torch.nonzero(vector).squeeze()
            weights = vector[indices]

            # Convert to dict
            sparse_vector = {
                int(idx): float(weight)
                for idx, weight in zip(indices, weights)
                if weight > 0
            }

            return sparse_vector

        except Exception as e:
            logger.error(f"SPLADE encoding failed: {e}")
            return {}

    def compute_similarity(
        self, query_vec: Dict[int, float], doc_vec: Dict[int, float]
    ) -> float:
        """
        Compute dot product similarity between two sparse vectors.

        Args:
            query_vec: Query sparse vector
            doc_vec: Document sparse vector

        Returns:
            Similarity score
        """
        score = 0.0
        # Iterate over the smaller vector for efficiency
        if len(query_vec) < len(doc_vec):
            for idx, weight in query_vec.items():
                if idx in doc_vec:
                    score += weight * doc_vec[idx]
        else:
            for idx, weight in doc_vec.items():
                if idx in query_vec:
                    score += weight * query_vec[idx]

        return score
