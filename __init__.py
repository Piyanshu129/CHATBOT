"""
Multi-turn chatbot with RAG and memory.

A conversational AI bot that uses:
- LangChain for chain orchestration
- ChromaDB for long-term memory (vector store)
- Short-term memory for recent context
- Support for multiple LLM backends (Ollama, HuggingFace)
"""

from .config import BotConfig
from .interface import Chatbot

__version__ = "1.0.0"
__all__ = ["BotConfig", "Chatbot"]
