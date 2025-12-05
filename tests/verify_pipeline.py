"""Verification script for the advanced RAG pipeline."""

import logging
import sys
from pathlib import Path

# Add parent directory to path
bot_dir = Path(__file__).resolve().parent.parent.parent
if str(bot_dir) not in sys.path:
    sys.path.insert(0, str(bot_dir))

from bot_custom.config import BotConfig  # noqa: E402
from bot_custom.interface.chatbot import Chatbot  # noqa: E402
from bot_custom.utils import setup_logging  # noqa: E402


def verify_pipeline():
    """Verify the advanced RAG pipeline."""
    setup_logging(level="DEBUG")
    logger = logging.getLogger(__name__)

    logger.info("Starting pipeline verification...")

    try:
        # Initialize config
        config = BotConfig(
            llm_backend="ollama",  # Assuming ollama is available, or mock it
            enable_splade=False,  # Disable SPLADE for quick test if model not present, or True if we want to test it
            enable_reranking=False,  # Disable for speed/dependency check
            # We'll try to enable everything that doesn't require heavy downloads if possible
            # But for safety, let's keep defaults but be aware of model loading
        )

        # Initialize chatbot
        logger.info("Initializing Chatbot...")
        bot = Chatbot(config)

        # Test query
        query = "What is the capital of France?"
        logger.info(f"Testing query: {query}")

        response = bot.chat(query, stream=False)

        logger.info(f"Response received: {response}")
        logger.info("Verification successful!")

    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    verify_pipeline()
