"""Context assembly module for intelligent context window management."""

import logging
from typing import List
import tiktoken
from langchain_core.documents import Document

from ..config import BotConfig

logger = logging.getLogger(__name__)


class ContextAssembler:
    """Assembles context from short-term and long-term memory within token limits."""

    def __init__(self, config: BotConfig):
        """
        Initialize context assembler.

        Args:
            config: Bot configuration
        """
        self.config = config
        self.max_tokens = config.max_context_tokens
        self.stm_budget = config.stm_token_budget

        # Initialize tokenizer (using cl100k_base for general purpose)
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(
                f"Failed to load tiktoken encoder: {e}, using character approximation"
            )
            self.tokenizer = None

        logger.info(
            f"Initialized ContextAssembler (max_tokens={self.max_tokens}, stm_budget={self.stm_budget})"
        )

    def assemble(
        self,
        short_term_history: str,
        long_term_docs: List[Document],
        question: str = "",
    ) -> dict:
        """
        Assemble context from short-term and long-term memory.

        Args:
            short_term_history: Formatted short-term conversation history
            long_term_docs: Retrieved long-term documents
            question: User question (for token budget calculation)

        Returns:
            Dictionary with 'history', 'context', and 'token_info'
        """
        # Count tokens in question and reserve space
        question_tokens = self._count_tokens(question)
        available_tokens = self.max_tokens - question_tokens

        # Allocate tokens: STM gets priority up to budget
        stm_tokens = min(self._count_tokens(short_term_history), self.stm_budget)
        ltm_budget = available_tokens - stm_tokens

        # Truncate STM if needed
        history = self._truncate_to_fit(short_term_history, self.stm_budget)

        # Assemble LTM context within budget
        context = self._assemble_ltm_context(long_term_docs, ltm_budget)

        # Calculate final token usage
        history_tokens = self._count_tokens(history)
        context_tokens = self._count_tokens(context)
        total_tokens = question_tokens + history_tokens + context_tokens

        logger.debug(
            f"Context assembled: {total_tokens}/{self.max_tokens} tokens "
            f"(history={history_tokens}, context={context_tokens}, question={question_tokens})"
        )

        return {
            "history": history,
            "context": context,
            "token_info": {
                "total": total_tokens,
                "history": history_tokens,
                "context": context_tokens,
                "question": question_tokens,
                "max": self.max_tokens,
                "utilization": total_tokens / self.max_tokens,
            },
        }

    def _assemble_ltm_context(self, documents: List[Document], budget: int) -> str:
        """
        Assemble long-term memory context within token budget.

        Args:
            documents: List of documents
            budget: Token budget

        Returns:
            Formatted context string
        """
        if not documents or budget <= 0:
            return ""

        # Prioritize documents (already ranked by reranker/hybrid search)
        context_parts = []
        used_tokens = 0

        for i, doc in enumerate(documents):
            doc_text = doc.page_content
            doc_tokens = self._count_tokens(doc_text)

            # Check if document fits in budget
            if used_tokens + doc_tokens <= budget:
                context_parts.append(doc_text)
                used_tokens += doc_tokens
            else:
                # Try to fit truncated version
                remaining_budget = budget - used_tokens
                if remaining_budget > 50:  # Only if we have meaningful space
                    truncated = self._truncate_to_fit(doc_text, remaining_budget)
                    context_parts.append(truncated + "...")
                    used_tokens += self._count_tokens(truncated)
                break

        return "\n\n".join(context_parts)

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        if not text:
            return 0

        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception as e:
                logger.warning(f"Token counting failed: {e}, using approximation")

        # Fallback: rough approximation (1 token ≈ 4 characters)
        return len(text) // 4

    def _truncate_to_fit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens

        Returns:
            Truncated text
        """
        if not text:
            return ""

        current_tokens = self._count_tokens(text)

        if current_tokens <= max_tokens:
            return text

        # Binary search for optimal truncation point
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                truncated_tokens = tokens[:max_tokens]
                return self.tokenizer.decode(truncated_tokens)

        # Fallback: character-based truncation
        ratio = max_tokens / current_tokens
        chars_to_keep = int(len(text) * ratio * 0.95)  # 0.95 for safety margin
        return text[:chars_to_keep]

    def get_budget_info(self) -> dict:
        """
        Get information about token budgets.

        Returns:
            Dictionary with budget information
        """
        return {
            "max_context_tokens": self.max_tokens,
            "stm_budget": self.stm_budget,
            "ltm_budget_relative": self.max_tokens - self.stm_budget,
        }
