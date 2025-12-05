"""LangChain chain builders for RAG functionality."""

import logging
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnableLambda
from langchain_core.language_models import BaseLLM

from ..config import BotConfig
from .memory import MemoryManager
from .classifier import QueryClassifier
from .query_expander import QueryExpander
from .reranker import Reranker

from .hallucination_filter import HallucinationFilter
from .graph_builder import ContextGraphBuilder
from .generation import ChainOfVerification
from .retrieval_guard import RetrievalGuard
from .cache_manager import CacheManager

from .context_assembler import ContextAssembler
from ..utils.timing import TimingContext

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

        # Initialize production-ready components
        self.classifier = (
            QueryClassifier(config.classifier_type, llm=self.llm)
            if config.use_query_classification
            else None
        )
        self.expander = (
            QueryExpander(config, llm) if config.enable_query_expansion else None
        )
        self.reranker = Reranker(config) if config.enable_reranking else None

        # Advanced RAG Components
        self.hallucination_filter = HallucinationFilter(config, llm)
        self.graph_builder = ContextGraphBuilder(config, llm)
        self.cov_generator = ChainOfVerification(config, llm)
        self.retrieval_guard = RetrievalGuard(config)
        self.cache_manager = CacheManager()

        self.context_assembler = ContextAssembler(config)

    def build_rag_chain(self) -> Runnable:
        """
        Build a RAG chain with memory integration.

        Returns:
            Runnable chain that takes a question and returns an answer
        """
        logger.info("Building RAG chain")

        setup_and_retrieval = RunnableParallel(
            {
                "context": itemgetter("question")
                | RunnableLambda(self.memory_manager.retrieve_long_term)
                | self.memory_manager.format_docs,
                "history": lambda x: self.memory_manager.get_short_term_history(),
                "question": itemgetter("question"),
            }
        )

        def timed_setup(x):
            with TimingContext("combined_memory"):
                return setup_and_retrieval.invoke(x)

        rag_chain = (
            RunnableLambda(timed_setup) | self.prompt | self.llm | StrOutputParser()
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
                "context": itemgetter("question")
                | RunnableLambda(self.memory_manager.retrieve_long_term)
                | self.memory_manager.format_docs,
                "docs": itemgetter("question")
                | RunnableLambda(self.memory_manager.retrieve_long_term),
                "history": lambda x: self.memory_manager.get_short_term_history(),
                "question": itemgetter("question"),
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

    def build_enhanced_rag_chain(self) -> Runnable:
        """
        Build an enhanced production-ready RAG chain with all advanced features.

        This chain includes:
        - Query classification
        - Multi-query expansion
        - Hybrid retrieval (cosine + BM25)
        - Cross-encoder reranking
        - Metadata filtering
        - Smart context assembly

        Returns:
            Runnable chain that takes a question and returns an answer
        """
        logger.info("Building enhanced RAG chain with production features")

        def enhanced_retrieval(x):
            """Full enhanced retrieval pipeline."""
            question = x.get("question", "")

            # Step 1: Query Classification
            classification = None
            if self.classifier:
                with TimingContext("query_classification"):
                    # Check cache first
                    cache_key = f"cls_{question}"
                    classification = self.cache_manager.get(cache_key)

                    if not classification:
                        classification = self.classifier.classify(question)
                        self.cache_manager.set(cache_key, classification)

                    logger.info(f"Query classified as: {classification.category.value}")

            # Step 3: Retrieval Guard check
            should_retrieve = self.retrieval_guard.should_retrieve(
                question, classification.category
            )

            if not should_retrieve:
                logger.info("Retrieval skipped by guardrails")
                return {
                    "context": "",
                    "history": self.memory_manager.get_short_term_history(),
                    "question": question,
                    "skip_retrieval": True,
                }

            # Step 4: Query Expansion (only if needed)
            queries = [question]
            if (
                self.expander and len(question.split()) > 4
            ):  # Skip expansion for short queries
                with TimingContext("query_expansion"):
                    # Check cache
                    cache_key = f"exp_{question}"
                    expanded = self.cache_manager.get(cache_key)

                    if not expanded:
                        expanded = self.expander.expand(question)
                        self.cache_manager.set(cache_key, expanded)

                    queries.extend(expanded)
                    logger.info(f"Generated {len(queries)} query variants")

            # Step 5: Hybrid Retrieval (Parallel)
            # Use the category from classification for routing
            category_filter = classification.category.value if classification else None

            with TimingContext("hybrid_retrieval"):
                # Pass category to retrieval for routing
                # retrieve_hybrid handles the fallback to simple retrieval if hybrid is disabled
                docs = self.memory_manager.retrieve_hybrid(
                    queries, category=category_filter
                )
                logger.info(
                    f"DEBUG: Retrieved docs (category={category_filter}): {[doc.page_content[:50] for doc in docs]}"
                )

            logger.debug(f"Retrieved {len(docs)} documents from hybrid search")

            # Step 5: Reranking
            if self.reranker and len(docs) > self.config.rerank_top_k:
                with TimingContext("reranking"):
                    docs = self.reranker.rerank(
                        question, docs, top_k=self.config.rerank_top_k
                    )
                    logger.info(
                        f"DEBUG: After reranking (top {self.config.rerank_top_k}): {[doc.page_content[:50] for doc in docs]}"
                    )

            # Step 5.5: Hallucination Filtering (Self-RAG)
            with TimingContext("hallucination_filtering"):
                docs = self.hallucination_filter.filter_documents(question, docs)

            # Step 5.6: Context Graph Building (GraphRAG)
            graph_summary = ""
            with TimingContext("graph_building"):
                graph_data = self.graph_builder.build_graph(docs)
                graph_summary = graph_data.get("summary", "")
                if graph_summary:
                    logger.info("DEBUG: Added graph summary to context")

            # Step 6: Context assembly
            short_term_history = self.memory_manager.get_short_term_history()

            with TimingContext("context_assembly"):
                assembled = self.context_assembler.assemble(
                    short_term_history=short_term_history,
                    long_term_docs=docs,
                    question=question,
                )

                # Append graph summary to context if available
                if graph_summary:
                    assembled["context"] += (
                        f"\n\nKnowledge Graph Summary:\n{graph_summary}"
                    )

            logger.info(
                f"DEBUG: Context assembled: {assembled['token_info']['total']}/{assembled['token_info']['max']} tokens"
            )
            logger.info(
                f"DEBUG: Final context being sent to LLM:\n{assembled['context']}"
            )

            # Calculate retrieval score (max of RRF scores or fallback)
            retrieval_score = 0.0
            if docs:
                scores = [doc.metadata.get("rrf_score", 0) for doc in docs]
                if not scores or max(scores) == 0:
                    # Fallback if no RRF scores (e.g. only cosine)
                    scores = [doc.metadata.get("score", 0) for doc in docs]

                if scores:
                    retrieval_score = max(scores)

            logger.info(f"Retrieval score: {retrieval_score:.4f}")

            return {
                "context": assembled["context"],
                "history": assembled["history"],
                "question": question,
                "category": category_filter,
                "retrieval_score": retrieval_score,
            }

        # Build the chain with CoV
        def generate_with_cov(x):
            return self.cov_generator.generate(
                question=x["question"],
                context=x["context"],
                category=x.get("category"),
                retrieval_score=x.get("retrieval_score", 1.0),
            )

        enhanced_chain = RunnableLambda(enhanced_retrieval) | RunnableLambda(
            generate_with_cov
        )

        logger.info(
            "Enhanced RAG chain built successfully (with CoV, GraphRAG, Self-RAG)"
        )
        return enhanced_chain
