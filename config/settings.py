"""Configuration settings for the chatbot."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BotConfig:
    """Configuration for the chatbot application."""
    
    # LLM Settings
    llm_backend: str = "ollama"  # Options: "ollama", "huggingface"
    ollama_model: str = "llama3:8b-instruct-q5_K_M"
    huggingface_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Generation Parameters
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.15
    do_sample: bool = True
    
    # Memory Settings
    chroma_persist_dir: str = "./chroma_chat_memory"
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_k: int = 3  # Number of documents to retrieve
    short_term_memory_size: int = 4  # Number of recent conversations to keep
    
    # Prompt Template
    prompt_template: str = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Use the following context and conversation history to answer the question.
If the question refers to previous messages (e.g. "what did I just say?"), rely on the Conversation History.
If the question requires knowledge from documents, rely on the Context.
Always combine both sources to provide a complete answer.

Conversation History:
{history}

Context:
{context}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    # Device Settings
    device_map: str = "auto"
    torch_dtype: str = "float16"
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> "BotConfig":
        """Create configuration from environment variables."""
        return cls(
            llm_backend=os.getenv("BOT_LLM_BACKEND", "ollama"),
            ollama_model=os.getenv("BOT_OLLAMA_MODEL", "llama3:8b-instruct-q5_K_M"),
            huggingface_model=os.getenv("BOT_HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
            chroma_persist_dir=os.getenv("BOT_CHROMA_DIR", "./chroma_chat_memory"),
            embedding_model=os.getenv("BOT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            log_level=os.getenv("BOT_LOG_LEVEL", "INFO"),
        )
    
    def validate(self) -> None:
        """Validate configuration settings."""
        if self.llm_backend not in ["ollama", "huggingface"]:
            raise ValueError(f"Invalid LLM backend: {self.llm_backend}. Must be 'ollama' or 'huggingface'")
        
        if self.short_term_memory_size < 1:
            raise ValueError("short_term_memory_size must be at least 1")
        
        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be at least 1")
        
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
