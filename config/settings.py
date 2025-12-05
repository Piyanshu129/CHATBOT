"""Configuration settings for the chatbot."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BotConfig:
    """Configuration for the chatbot application."""

    # LLM Settings
    llm_backend: str = "ollama"  # Options: "ollama", "huggingface", "vllm"
    ollama_model: str = "llama3:8b-instruct-q5_K_M"
    huggingface_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    vllm_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    vllm_url: str = "http://localhost:8000/v1"

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
    prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant with access to a long-term memory (Context).
    Your goal is to answer the user's question using the provided information.

    CRITICAL INSTRUCTIONS:
    1. "Context" contains your Long-Term Memory (past conversations, user details, facts).
    2. "Conversation History" contains the Short-Term Memory (current session).
    3. If the user asks about personal details (e.g., "what is my favorite color?", "what car do I like?"), you MUST check the "Context" section.
    4. Do NOT say "I don't know" or "we haven't discussed this" if the answer is present in the "Context".
    5. Trust the "Context" as the source of truth for past interactions.

    Conversation History:
    {history}

    Context (Long-Term Memory):
    {context}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    # Device Settings
    device_map: str = "auto"
    torch_dtype: str = "float16"

    # Logging
    log_level: str = "INFO"

    # ==== Production-Ready RAG Enhancements ====

    # Query Classification
    use_query_classification: bool = True
    classifier_type: str = "llm"  # Options: "rule_based", "llm"

    # Multi-Query Expansion
    enable_query_expansion: bool = True
    num_query_variants: int = 4
    expansion_cache_size: int = 100
    expansion_model: str = "ollama"  # Use same as llm_backend or specify

    # Hybrid Retrieval
    enable_hybrid_retrieval: bool = True
    hybrid_retrieval_k: int = 20  # Candidates before reranking

    # Fusion Weights (must sum to approx 1.0 for clarity, though RRF is rank-based)
    bm25_weight: float = 0.3
    cosine_weight: float = 0.4
    splade_weight: float = 0.3

    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    # SPLADE Settings
    enable_splade: bool = True
    splade_model_id: str = "naver/splade-cocondenser-ensembledistil"

    # Reranking
    enable_reranking: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 3

    # Context Assembly
    max_context_tokens: int = 2048
    stm_token_budget: int = 512

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Create configuration from environment variables."""
        return cls(
            llm_backend=os.getenv("BOT_LLM_BACKEND", "ollama"),
            ollama_model=os.getenv("BOT_OLLAMA_MODEL", "llama3:8b-instruct-q5_K_M"),
            huggingface_model=os.getenv(
                "BOT_HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"
            ),
            chroma_persist_dir=os.getenv("BOT_CHROMA_DIR", "./chroma_chat_memory"),
            embedding_model=os.getenv("BOT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            log_level=os.getenv("BOT_LOG_LEVEL", "INFO"),
            vllm_model=os.getenv(
                "BOT_VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"
            ),
            vllm_url=os.getenv("BOT_VLLM_URL", "http://localhost:8000/v1"),
        )

    def validate(self) -> None:
        """Validate configuration settings."""
        if self.llm_backend not in ["ollama", "huggingface", "vllm"]:
            raise ValueError(
                f"Invalid LLM backend: {self.llm_backend}. Must be 'ollama', 'huggingface', or 'vllm'"
            )

        if self.short_term_memory_size < 1:
            raise ValueError("short_term_memory_size must be at least 1")

        if self.retrieval_k < 1:
            raise ValueError("retrieval_k must be at least 1")

        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
