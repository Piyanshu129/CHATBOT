# Multi-turn Chatbot with RAG and Memory

A conversational AI chatbot built with LangChain that features both short-term and long-term memory capabilities using ChromaDB for vector storage.

## Features

- 🧠 **Dual Memory System**
  - Short-term memory: Recent conversation context (configurable, default 4 turns)
  - Long-term memory: Vector-based storage using ChromaDB for historical conversations
  
- 🤖 **Multiple LLM Backends**
  - Ollama (default, lightweight and fast)
  - HuggingFace Transformers (for custom models)
  
- 🔄 **RAG Capability**
  - Retrieval-Augmented Generation using conversation history
  - Context-aware responses based on past interactions
  
- 📡 **Streaming Support**
  - Real-time response generation
  - Token-by-token output for better UX
  
- 🎯 **Modular Architecture**
  - Clean separation of concerns
  - Easy to extend and customize
  - Well-documented codebase

## Installation

1. **Clone or navigate to this directory**

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **For Ollama backend (recommended)**
   - Install Ollama from [ollama.ai](https://ollama.ai)
   - Pull a model:
   ```bash
   ollama pull llama3:8b-instruct-q5_K_M
   ```

## Usage

### Interactive Chat (CLI)

Start an interactive chat session:

```bash
python main.py
```

With custom settings:

```bash
# Use different Ollama model
python main.py --model llama3:70b

# Use HuggingFace backend
python main.py --backend huggingface --model meta-llama/Meta-Llama-3-8B-Instruct

# Adjust memory size
python main.py --memory-size 10

# Custom ChromaDB directory
python main.py --chroma-dir ./my_custom_memory
```

### One-shot Query

Process a single query without interactive mode:

```bash
python main.py --query "What is machine learning?"
```

### Programmatic Usage

```python
from bot import Chatbot, BotConfig

# Use default configuration
bot = Chatbot()

# Or customize configuration
config = BotConfig(
    llm_backend="ollama",
    ollama_model="llama3:8b-instruct-q5_K_M",
    short_term_memory_size=6,
    temperature=0.8
)
bot = Chatbot(config)

# Interactive mode
bot.interactive_chat()

# Or programmatic chat
response = bot.chat("Hello, how are you?")
print(response)

# Stream responses
for chunk in bot.stream_chat("Tell me a story"):
    print(chunk, end="", flush=True)

# Chat with source documents
result = bot.chat_with_sources("What did we discuss earlier?")
print(result['answer'])
print(f"Used {len(result['docs'])} source documents")
```

## Architecture

```
bot/
├── config/             # Configuration management
│   ├── __init__.py
│   └── settings.py     # BotConfig dataclass
├── core/               # Core functionality
│   ├── __init__.py
│   ├── llm_models.py   # LLM factory for different backends
│   ├── memory.py       # Memory management (ChromaDB + short-term)
│   └── chains.py       # LangChain chain builders
├── interface/          # User interfaces
│   ├── __init__.py
│   ├── chatbot.py      # Main Chatbot class
│   └── cli.py          # Command-line interface
├── utils/              # Utilities
│   ├── __init__.py
│   └── helpers.py      # Helper functions
├── __init__.py         # Package initialization
├── main.py             # Entry point
└── requirements.txt    # Dependencies
```

## Configuration

### Environment Variables

You can configure the bot using environment variables:

```bash
export BOT_LLM_BACKEND=ollama
export BOT_OLLAMA_MODEL=llama3:8b-instruct-q5_K_M
export BOT_CHROMA_DIR=./chroma_chat_memory
export BOT_LOG_LEVEL=INFO
```

### Configuration Options

The `BotConfig` class supports the following options:

- **LLM Settings**
  - `llm_backend`: "ollama" or "huggingface"
  - `ollama_model`: Ollama model name
  - `huggingface_model`: HuggingFace model ID
  
- **Generation Parameters**
  - `max_new_tokens`: Maximum tokens to generate (default: 1024)
  - `temperature`: Sampling temperature (default: 0.7)
  - `top_p`: Nucleus sampling parameter (default: 0.95)
  - `repetition_penalty`: Penalty for repetition (default: 1.15)
  
- **Memory Settings**
  - `chroma_persist_dir`: ChromaDB storage directory
  - `embedding_model`: Sentence transformer model for embeddings
  - `retrieval_k`: Number of documents to retrieve (default: 3)
  - `short_term_memory_size`: Recent conversations to keep (default: 4)

## Commands (Interactive Mode)

While in interactive chat:
- `exit` or `quit`: Exit the chat
- `clear`: Clear short-term memory
- `stats`: Show memory statistics

## How It Works

1. **User Input**: User provides a question or message
2. **Context Retrieval**: 
   - Recent conversations retrieved from short-term memory
   - Relevant past conversations retrieved from ChromaDB vector store
3. **Prompt Construction**: Context and history combined into a prompt
4. **LLM Generation**: Response generated using selected LLM backend
5. **Memory Update**: Interaction saved to both memories for future retrieval

## Extending the Bot

### Add a New LLM Backend

Edit `core/llm_models.py` and add a new method to `LLMFactory`:

```python
@staticmethod
def _create_custom_llm(config: BotConfig):
    # Your implementation
    return llm_instance
```

### Customize the Prompt

Modify the `prompt_template` in `BotConfig` or update it programmatically:

```python
bot.chain_builder.update_prompt_template("Your custom template here")
```

### Add New Memory Types

Extend the `MemoryManager` class in `core/memory.py` to support additional memory backends.

## Troubleshooting

### Ollama Connection Issues
- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`

### Out of Memory (HuggingFace)
- Use Ollama backend instead (much more memory efficient)
- Reduce `max_new_tokens`
- Use quantized models

### ChromaDB Errors
- Delete and reinitialize: `rm -rf chroma_chat_memory`
- Check disk space and permissions

## License

This project is open source and available for modification and distribution.

## Credits

Built with:
- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Ollama](https://ollama.ai)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
