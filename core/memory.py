"""Memory management for the chatbot using ChromaDB and short-term memory."""

import logging
from collections import deque
from typing import List, Deque
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..config import BotConfig

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages both short-term and long-term (vector store) memory."""
    
    def __init__(self, config: BotConfig):
        """
        Initialize memory manager.
        
        Args:
            config: Bot configuration object
        """
        self.config = config
        self.short_term_memory: Deque[str] = deque(maxlen=config.short_term_memory_size)
        self._vectorstore: Chroma = None
        self._retriever: BaseRetriever = None
        self._embeddings: HuggingFaceEmbeddings = None
        
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self) -> None:
        """Initialize the vector store and retriever."""
        try:
            logger.info(f"Initializing embeddings with model: {self.config.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )
            
            logger.info(f"Initializing ChromaDB at: {self.config.chroma_persist_dir}")
            self._vectorstore = Chroma(
                persist_directory=self.config.chroma_persist_dir,
                embedding_function=self._embeddings
            )
            
            self._retriever = self._vectorstore.as_retriever(
                search_kwargs={"k": self.config.retrieval_k}
            )
            
            logger.info("Vector store initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}") from e
    
    @property
    def retriever(self) -> BaseRetriever:
        """Get the retriever instance."""
        return self._retriever
    
    @property
    def vectorstore(self) -> Chroma:
        """Get the vector store instance."""
        return self._vectorstore
    
    def save_interaction(self, user_input: str, bot_response: str) -> None:
        """
        Save a conversation interaction to both short-term and long-term memory.
        
        Args:
            user_input: User's message
            bot_response: Bot's response
        """
        # Format the interaction
        interaction_text = f"Human: {user_input}\nAssistant: {bot_response}"
        
        # Add to short-term memory
        self.short_term_memory.append(interaction_text)
        logger.debug(f"Added to short-term memory. Current size: {len(self.short_term_memory)}")
        
        # Add to vector store (long-term memory)
        try:
            doc = Document(
                page_content=interaction_text,
                metadata={"type": "interaction"}
            )
            self._vectorstore.add_documents([doc])
            logger.debug("Saved interaction to vector store")
        except Exception as e:
            logger.error(f"Failed to save interaction to vector store: {e}")
            # Don't raise - allow chatbot to continue even if vector store fails
    
    def get_short_term_history(self) -> str:
        """
        Get formatted short-term conversation history.
        
        Returns:
            Formatted string of recent conversations
        """
        return "\n".join(self.short_term_memory)
    
    def format_docs(self, docs: List[Document]) -> str:
        """
        Format retrieved documents into a single string.
        
        Args:
            docs: List of retrieved documents
            
        Returns:
            Formatted string of document contents
        """
        return "\n\n".join(doc.page_content for doc in docs)
    
    def clear_short_term_memory(self) -> None:
        """Clear the short-term memory."""
        self.short_term_memory.clear()
        logger.info("Short-term memory cleared")
    
    def get_memory_stats(self) -> dict:
        """
        Get statistics about memory usage.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            collection = self._vectorstore._collection
            vector_count = collection.count()
        except Exception:
            vector_count = "unknown"
        
        return {
            "short_term_size": len(self.short_term_memory),
            "short_term_capacity": self.short_term_memory.maxlen,
            "vector_store_documents": vector_count,
        }
