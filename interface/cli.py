"""Command-line interface for the chatbot."""

import argparse
import sys
import logging
from typing import Optional

from ..config.settings import BotConfig
from ..utils import setup_logging
from .chatbot import Chatbot

logger = logging.getLogger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Multi-turn chatbot with RAG and memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive chat with Ollama
  python -m bot.interface.cli
  
  # Use HuggingFace backend
  python -m bot.interface.cli --backend huggingface

  # Use vLLM backend
  python -m bot.interface.cli --backend vllm --vllm-url http://localhost:8000/v1
  
  # One-shot query
  python -m bot.interface.cli --query "What is the capital of France?"
  
  # Custom configuration
  python -m bot.interface.cli --model llama3:70b --memory-size 10
        """,
    )

    parser.add_argument(
        "--backend",
        choices=["ollama", "huggingface", "vllm"],
        default="ollama",
        help="LLM backend to use (default: ollama)",
    )

    parser.add_argument(
        "--model", help="Model name/ID to use (overrides default for backend)"
    )

    parser.add_argument(
        "--chroma-dir",
        default="./chroma_chat_memory",
        help="Directory for ChromaDB persistence (default: ./chroma_chat_memory)",
    )

    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000/v1",
        help="URL for vLLM server (default: http://localhost:8000/v1)",
    )

    parser.add_argument(
        "--memory-size",
        type=int,
        default=4,
        help="Number of recent conversations to keep in short-term memory (default: 4)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    parser.add_argument(
        "--query", help="Single query to process (non-interactive mode)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument("--log-file", help="Optional log file path")

    return parser


def main(args: Optional[list] = None) -> int:
    """
    Main CLI entry point.

    Args:
        args: Command-line arguments (for testing). If None, uses sys.argv

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_argument_parser()
    parsed_args = parser.parse_args(args)

    # Setup logging
    setup_logging(level=parsed_args.log_level, log_file=parsed_args.log_file)

    try:
        # Create configuration
        config = BotConfig(
            llm_backend=parsed_args.backend,
            chroma_persist_dir=parsed_args.chroma_dir,
            short_term_memory_size=parsed_args.memory_size,
            temperature=parsed_args.temperature,
            log_level=parsed_args.log_level,
            vllm_url=parsed_args.vllm_url,
        )

        # Override model if specified
        if parsed_args.model:
            if parsed_args.backend == "ollama":
                config.ollama_model = parsed_args.model
            elif parsed_args.backend == "huggingface":
                config.huggingface_model = parsed_args.model
            elif parsed_args.backend == "vllm":
                config.vllm_model = parsed_args.model

        # Initialize chatbot
        logger.info("Initializing chatbot...")
        bot = Chatbot(config)

        # Handle query mode or interactive mode
        if parsed_args.query:
            # One-shot query
            logger.info(f"Processing query: {parsed_args.query}")
            response = bot.chat(parsed_args.query, stream=False)
            print(f"\nBot: {response}\n")
        else:
            # Interactive mode
            bot.interactive_chat()

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
