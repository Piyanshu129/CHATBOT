"""LangChain chain builders for RAG functionality."""

import logging
from typing import Any, Union
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_core.language_models import BaseLLM

from ..config import BotConfig
from .memory import MemoryManager

logger = logging.getLogger(__name__)


class ChainBuilder:
    """Builder class for creating LangChain chains."""
    
    def __init__(self, config: BotConfig, llm: BaseLLM, memory_manager: MemoryManager):
        """
        Initialize chain builder.
        
        Args:
            config: Bot configuration object
            llm: Language model instance
            memory_manager: Memory manager instance
        """
        self.config = config
        self.llm = llm
        self.memory_manager = memory_manager
        self.prompt = ChatPromptTemplate.from_template(config.prompt_template)
    
    def build_rag_chain(self) -> Runnable:
        """
        Build a RAG chain with memory integration.
        
        Returns:
            Runnable chain that takes a question and returns an answer
        """
        logger.info("Building RAG chain")
        
        rag_chain = (
            {
                "context": itemgetter("question") | self.memory_manager.retriever | self.memory_manager.format_docs,
                "history": lambda x: self.memory_manager.get_short_term_history(),
                "question": itemgetter("question")
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        logger.info("RAG chain built successfully")
        return rag_chain
    
    def build_rag_chain_with_sources(self) -> Runnable:
        """
        Build a RAG chain that returns both answer and source documents.
        
        Returns:
            Runnable chain that returns a dict with 'answer', 'docs', 'context', and 'question'
        """
        logger.info("Building RAG chain with sources")
        
        # Setup: Retrieve docs AND keep them
        setup_and_retrieval = RunnableParallel(
            {
                "context": itemgetter("question") | self.memory_manager.retriever | self.memory_manager.format_docs,
                "docs": itemgetter("question") | self.memory_manager.retriever,
                "history": lambda x: self.memory_manager.get_short_term_history(),
                "question": itemgetter("question")
            }
        )
        
        # Generate answer using the context
        rag_chain_with_source = setup_and_retrieval.assign(
            answer=self.prompt | self.llm | StrOutputParser()
        )
        
        logger.info("RAG chain with sources built successfully")
        return rag_chain_with_source
    
    def update_prompt_template(self, new_template: str) -> None:
        """
        Update the prompt template.
        
        Args:
            new_template: New prompt template string
        """
        logger.info("Updating prompt template")
        self.config.prompt_template = new_template
        self.prompt = ChatPromptTemplate.from_template(new_template)
        logger.info("Prompt template updated")
