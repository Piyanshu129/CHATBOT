"""Main chatbot interface class."""

import logging
from typing import Iterator, Optional, Dict, Any

from ..config import BotConfig
from ..core import LLMFactory, MemoryManager, ChainBuilder
from ..utils import filter_special_tokens

logger = logging.getLogger(__name__)


class Chatbot:
    """Main chatbot class that integrates all components."""
    
    def __init__(self, config: Optional[BotConfig] = None):
        """
        Initialize the chatbot.
        
        Args:
            config: Bot configuration. If None, uses default config.
        """
        self.config = config or BotConfig()
        self.config.validate()
        
        logger.info("Initializing Chatbot...")
        
        # Initialize components
        self.llm = LLMFactory.create_llm(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.chain_builder = ChainBuilder(self.config, self.llm, self.memory_manager)
        
        # Build the main chain
        self.chain = self.chain_builder.build_rag_chain()
        
        logger.info("Chatbot initialized successfully")
    
    def chat(self, user_input: str, stream: bool = True) -> str:
        """
        Process user input and generate a response.
        
        Args:
            user_input: User's message
            stream: Whether to stream the response
            
        Returns:
            Bot's response as a complete string
        """
        if not user_input.strip():
            logger.warning("Empty user input received")
            return "Please provide a message."
        
        if stream:
            # Collect all chunks for streaming
            full_response = ""
            for chunk in self.stream_chat(user_input):
                full_response += chunk
            return full_response
        else:
            # Non-streaming response
            try:
                response = self.chain.invoke({"question": user_input})
                response = filter_special_tokens(response)
                
                # Save to memory
                self.memory_manager.save_interaction(user_input, response)
                
                return response
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                return f"I'm sorry, I encountered an error: {str(e)}"
    
    def stream_chat(self, user_input: str) -> Iterator[str]:
        """
        Process user input and stream the response.
        
        Args:
            user_input: User's message
            
        Yields:
            Response chunks
        """
        full_response = ""
        
        try:
            for chunk in self.chain.stream({"question": user_input}):
                # Filter special tokens
                chunk = filter_special_tokens(chunk)
                full_response += chunk
                yield chunk
            
            # Save interaction after streaming completes
            self.memory_manager.save_interaction(user_input, full_response)
            
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            yield f"\nI'm sorry, I encountered an error: {str(e)}"
    
    def chat_with_sources(self, user_input: str) -> Dict[str, Any]:
        """
        Chat and return both answer and source documents.
        
        Args:
            user_input: User's message
            
        Returns:
            Dictionary with 'answer', 'docs', 'context', and 'question' keys
        """
        try:
            chain_with_sources = self.chain_builder.build_rag_chain_with_sources()
            result = chain_with_sources.invoke({"question": user_input})
            
            # Filter answer
            result['answer'] = filter_special_tokens(result['answer'])
            
            # Save to memory
            self.memory_manager.save_interaction(user_input, result['answer'])
            
            return result
            
        except Exception as e:
            logger.error(f"Error in chat_with_sources: {e}")
            return {
                "answer": f"I'm sorry, I encountered an error: {str(e)}",
                "docs": [],
                "context": "",
                "question": user_input
            }
    
    def clear_short_term_memory(self) -> None:
        """Clear short-term conversation history."""
        self.memory_manager.clear_short_term_memory()
        logger.info("Short-term memory cleared")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        return self.memory_manager.get_memory_stats()
    
    def interactive_chat(self) -> None:
        """
        Start an interactive chat session in the terminal.
        """
        print("✅ Chatbot Ready! Type 'exit' or 'quit' to stop.")
        print("Commands:")
        print("  - 'clear': Clear short-term memory")
        print("  - 'stats': Show memory statistics")
        print()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == "clear":
                    self.clear_short_term_memory()
                    print("✓ Short-term memory cleared")
                    continue
                
                if user_input.lower() == "stats":
                    stats = self.get_memory_stats()
                    print(f"📊 Memory Statistics:")
                    for key, value in stats.items():
                        print(f"  - {key}: {value}")
                    continue
                
                # Generate response
                print("Bot: ", end="", flush=True)
                
                for chunk in self.stream_chat(user_input):
                    print(chunk, end="", flush=True)
                
                print()  # New line after response
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                logger.error(f"Error in interactive chat: {e}")
                print(f"\nError: {e}")
