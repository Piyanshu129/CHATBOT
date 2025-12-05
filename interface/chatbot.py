"""Main chatbot interface class."""

import logging
from typing import Iterator, Optional, Dict, Any

from ..config import BotConfig
from ..core import LLMFactory, MemoryManager, ChainBuilder, MemoryEncoder
from ..core.classifier import QueryClassifier
from ..utils import filter_special_tokens
from ..utils.timing import TimingManager, TimingContext

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

        # self.classifier = QueryClassifier()
        self.classifier = QueryClassifier(
            classifier_type=self.config.classifier_type, llm=self.llm
        )

        self.memory_manager = MemoryManager(self.config)
        self.chain_builder = ChainBuilder(self.config, self.llm, self.memory_manager)
        self.memory_encoder = MemoryEncoder(self.config, self.llm)

        # Build the main chain - use enhanced chain with production features
        self.chain = self.chain_builder.build_enhanced_rag_chain()

        self.timing_manager = TimingManager()

        logger.info("Chatbot initialized successfully with enhanced RAG pipeline")

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
                # Reset timing for this query (except model_loading)
                model_loading = self.timing_manager.stats.model_loading
                self.timing_manager.reset()
                self.timing_manager.stats.model_loading = model_loading

                # response = self.chain.invoke({"question": user_input})
                # response = filter_special_tokens(response)

                # # Save to memory
                # self.memory_manager.save_interaction(user_input, response)
                # Classify the query
                classification = self.classifier.classify(user_input)
                category = classification.category.value

                # Generate response
                response = self.chain.invoke({"question": user_input})
                response = filter_special_tokens(response)

                # Save with the correct category
                self.memory_manager.save_interaction(
                    user_input=user_input, bot_response=response, category=category
                )

                # Encode and save structured memories
                memories = self.memory_encoder.encode_interaction(user_input, response)
                for mem in memories:
                    self.memory_manager.save_memory_fact(
                        mem["content"], mem["category"]
                    )
                # --- new code ends here ---
                # Log timing stats
                stats = self.timing_manager.get_stats()
                logger.info(f"Timing Stats: {stats}")

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
            # Reset timing for this query (except model_loading)
            model_loading = self.timing_manager.stats.model_loading
            self.timing_manager.reset()
            self.timing_manager.stats.model_loading = model_loading

            # Start TTFT timer
            start_time = 0
            first_token_received = False

            with TimingContext("ttft") as ttft_ctx:
                start_time = ttft_ctx.start_time
                for chunk in self.chain.stream({"question": user_input}):
                    if not first_token_received:
                        # TTFT is measured by the context manager automatically when we exit,
                        # but here we are inside the loop.
                        # We need to manually set the duration for TTFT or use the context manager differently.
                        # Actually, TimingContext measures block execution.
                        # We want time from start of stream call until first chunk.
                        # But `self.chain.stream` is a generator. The time until first yield is TTFT.
                        pass

                    # Filter special tokens
                    chunk = filter_special_tokens(chunk)

                    if not first_token_received and chunk:
                        # This is the first token/chunk
                        first_token_received = True
                        # We can't exit the context manager here easily.
                        # Let's manually record TTFT using the manager.
                        import time

                        duration = time.time() - start_time
                        self.timing_manager.stats.ttft = duration
                        logger.debug(f"Timing - ttft: {duration:.4f}s")

                    full_response += chunk
                    yield chunk

            # Log timing stats
            stats = self.timing_manager.get_stats()
            logger.info(f"Timing Stats: {stats}")

            # Save interaction after streaming completes
            # self.memory_manager.save_interaction(user_input, full_response)
            # Classify before saving
            classification = self.classifier.classify(user_input)
            category = classification.category.value

            self.memory_manager.save_interaction(
                user_input=user_input, bot_response=full_response, category=category
            )

            # Encode and save structured memories
            memories = self.memory_encoder.encode_interaction(user_input, full_response)
            for mem in memories:
                self.memory_manager.save_memory_fact(mem["content"], mem["category"])

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
            result["answer"] = filter_special_tokens(result["answer"])

            # Save to memory
            # self.memory_manager.save_interaction(user_input, result["answer"])

            classification = self.classifier.classify(user_input)
            category = classification.category.value

            self.memory_manager.save_interaction(
                user_input=user_input, bot_response=result["answer"], category=category
            )

            # Encode and save structured memories
            memories = self.memory_encoder.encode_interaction(
                user_input, result["answer"]
            )
            for mem in memories:
                self.memory_manager.save_memory_fact(mem["content"], mem["category"])

            return result

        except Exception as e:
            logger.error(f"Error in chat_with_sources: {e}")
            return {
                "answer": f"I'm sorry, I encountered an error: {str(e)}",
                "docs": [],
                "context": "",
                "question": user_input,
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
                    print("📊 Memory Statistics:")
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
