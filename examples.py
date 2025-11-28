"""Example usage of the chatbot package."""

from bot import Chatbot, BotConfig


def example_basic_usage():
    """Example: Basic chatbot usage with default settings."""
    print("=== Example 1: Basic Usage ===\n")
    
    # Create chatbot with default configuration (uses Ollama)
    bot = Chatbot()
    
    # Simple chat
    response = bot.chat("Hello! What can you help me with?", stream=False)
    print(f"User: Hello! What can you help me with?")
    print(f"Bot: {response}\n")


def example_custom_config():
    """Example: Custom configuration."""
    print("=== Example 2: Custom Configuration ===\n")
    
    # Create custom configuration
    config = BotConfig(
        llm_backend="ollama",
        ollama_model="llama3:8b-instruct-q5_K_M",
        short_term_memory_size=6,  # Keep more context
        temperature=0.8,  # More creative responses
        chroma_persist_dir="./custom_memory"
    )
    
    bot = Chatbot(config)
    response = bot.chat("Tell me something interesting")
    print(f"Bot: {response}\n")


def example_streaming():
    """Example: Streaming responses."""
    print("=== Example 3: Streaming ===\n")
    
    bot = Chatbot()
    
    print("User: Write a short poem about AI")
    print("Bot: ", end="", flush=True)
    
    for chunk in bot.stream_chat("Write a short poem about AI"):
        print(chunk, end="", flush=True)
    
    print("\n")


def example_with_sources():
    """Example: Chat with source documents."""
    print("=== Example 4: Chat with Sources ===\n")
    
    bot = Chatbot()
    
    # First, have a conversation
    bot.chat("My favorite color is blue", stream=False)
    bot.chat("I like to play chess", stream=False)
    
    # Now ask about previous conversation
    result = bot.chat_with_sources("What do you know about me?")
    
    print(f"Answer: {result['answer']}")
    print(f"\nUsed {len(result['docs'])} source documents:")
    for i, doc in enumerate(result['docs'], 1):
        print(f"  {i}. {doc.page_content[:100]}...")
    print()


def example_memory_management():
    """Example: Memory management."""
    print("=== Example 5: Memory Management ===\n")
    
    bot = Chatbot()
    
    # Add some conversations
    bot.chat("My name is Alice", stream=False)
    bot.chat("I work as a data scientist", stream=False)
    
    # Check memory stats
    stats = bot.get_memory_stats()
    print(f"Memory Stats: {stats}")
    
    # Clear short-term memory
    bot.clear_short_term_memory()
    print("Short-term memory cleared")
    
    stats = bot.get_memory_stats()
    print(f"Memory Stats after clear: {stats}\n")


def example_interactive():
    """Example: Interactive chat mode."""
    print("=== Example 6: Interactive Mode ===\n")
    print("Starting interactive chat mode...")
    print("(This will open an interactive session)\n")
    
    bot = Chatbot()
    bot.interactive_chat()


if __name__ == "__main__":
    # Run examples (comment out the ones you don't want to run)
    
    # example_basic_usage()
    # example_custom_config()
    # example_streaming()
    # example_with_sources()
    # example_memory_management()
    
    # Interactive mode (uncomment to use)
    example_interactive()
