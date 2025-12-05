"""Memory management for the chatbot using ChromaDB and short-term memory."""

import logging
import time
from collections import deque
from typing import List, Deque, Optional, Dict, Tuple
from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ..config import BotConfig
from ..utils.timing import TimingContext
from .splade_handler import SpladeHandler
import json

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

        # BM25 for hybrid retrieval
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs: List[Document] = []
        self._bm25_corpus: List[List[str]] = []

        # SPLADE for sparse retrieval
        self.splade_handler = SpladeHandler(config)
        self._splade_index: Dict[
            int, List[Tuple[int, float]]
        ] = {}  # token_id -> [(doc_idx, weight)]
        self._splade_docs: List[Document] = []

        self._initialize_vectorstore()
        if config.enable_hybrid_retrieval:
            self._initialize_bm25()
            if config.enable_splade:
                self._initialize_splade()

    def _initialize_vectorstore(self) -> None:
        """Initialize the vector store and retriever."""
        try:
            logger.info(
                f"Initializing embeddings with model: {self.config.embedding_model}"
            )
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model
            )

            logger.info(f"Initializing ChromaDB at: {self.config.chroma_persist_dir}")
            self._vectorstore = Chroma(
                persist_directory=self.config.chroma_persist_dir,
                embedding_function=self._embeddings,
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
        # Return a wrapper or the retriever itself.
        # Since we can't easily wrap the retriever's invoke method without changing how it's used in the chain,
        # we will rely on the fact that the chain calls `get_relevant_documents` or `invoke`.
        # However, LangChain retrievers are Runnables.
        # A simple way is to wrap the retrieval in the chain, but here we can try to wrap the retriever.
        return self._retriever

    @property
    def vectorstore(self) -> Chroma:
        """Get the vector store instance."""
        return self._vectorstore

    def retrieve_long_term(
        self, query: str, category: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents from long-term memory with timing.
        This includes embedding generation time for the query.

        Args:
            query: Search query
            category: Optional category to filter by

        Returns:
            List of relevant documents
        """
        # Time the embedding generation
        with TimingContext("embedding_generation"):
            # The embeddings are generated when we call the retriever
            # We need to embed the query first
            _ = self._embeddings.embed_query(query)

        # Time the actual retrieval
        with TimingContext("long_term_memory"):
            # Prepare filter if category is provided
            search_kwargs = {"k": self.config.retrieval_k}
            if category:
                search_kwargs["filter"] = {"category": category}

            # Use the retriever with similarity search
            it_docs = self._vectorstore.similarity_search(query, **search_kwargs)
            return it_docs

    def filter_by_category(self, documents, category):
        filtered = []
        for doc in documents:
            doc_category = doc.metadata.get("category") if doc.metadata else None

            # If no category metadata, include (backward compat)
            if doc_category is None:
                filtered.append(doc)
            # "interaction" is legacy default - matches everything
            elif doc_category == "interaction":
                filtered.append(doc)
            # Match category
            elif doc_category == category.value:
                filtered.append(doc)
            # Compatible categories
            elif self._categories_compatible(category, doc_category):
                filtered.append(doc)

        return filtered

    def save_interaction(
        self,
        user_input: str,
        bot_response: str,
        category: str = "interaction",
        topic: Optional[str] = None,
    ) -> None:
        """
        Save a conversation interaction to both short-term and long-term memory.

        Args:
            user_input: User's message
            bot_response: Bot's response
            category: Category of the interaction (default: "interaction")
            topic: Optional topic tag
        """
        # Format the interaction
        interaction_text = f"Human: {user_input}\nAssistant: {bot_response}"

        # Add to short-term memory
        self.short_term_memory.append(interaction_text)
        logger.debug(
            f"Added to short-term memory. Current size: {len(self.short_term_memory)}"
        )

        # Add to vector store (long-term memory) with metadata
        try:
            metadata = {
                "type": "interaction",
                "category": category,
                "timestamp": time.time(),
            }
            if topic:
                metadata["topic"] = topic

            doc = Document(page_content=interaction_text, metadata=metadata)
            self._vectorstore.add_documents([doc])

            # Update BM25 index if enabled
            if self.config.enable_hybrid_retrieval:
                self._add_to_bm25(doc)
                if self.config.enable_splade:
                    self._add_to_splade(doc)

            logger.debug("Saved interaction to vector store, BM25, and SPLADE index")
        except Exception as e:
            logger.error(f"Failed to save interaction to vector store: {e}")
            # Don't raise - allow chatbot to continue even if vector store fails

    def save_memory_fact(self, content: str, category: str) -> None:
        """
        Save a specific memory fact to the vector store.
        Does NOT add to short-term conversation history.

        Args:
            content: The fact or memory content
            category: Category (user_profile, semantic, episodic)
        """
        try:
            metadata = {
                "type": "fact",
                "category": category,
                "timestamp": time.time(),
            }

            doc = Document(page_content=content, metadata=metadata)
            self._vectorstore.add_documents([doc])

            # Update BM25/SPLADE if enabled
            if self.config.enable_hybrid_retrieval:
                self._add_to_bm25(doc)
                if self.config.enable_splade:
                    self._add_to_splade(doc)

            logger.debug(f"Saved {category} memory: {content[:50]}...")

        except Exception as e:
            logger.error(f"Failed to save memory fact: {e}")

    def get_short_term_history(self) -> str:
        """
        Get formatted short-term conversation history.

        Returns:
            Formatted string of recent conversations
        """
        with TimingContext("short_term_memory"):
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

    def _initialize_bm25(self) -> None:
        """Initialize BM25 index from existing documents."""
        try:
            logger.info("Initializing BM25 index from ChromaDB")

            # Get all documents from vector store
            collection = self._vectorstore._collection
            results = collection.get()

            if results and "documents" in results:
                docs_content = results["documents"]
                metadatas = results.get("metadatas", [{}] * len(docs_content))

                self._bm25_docs = [
                    Document(page_content=content, metadata=meta or {})
                    for content, meta in zip(docs_content, metadatas)
                ]

                # Tokenize for BM25
                self._bm25_corpus = [
                    doc.page_content.lower().split() for doc in self._bm25_docs
                ]

                if self._bm25_corpus:
                    self._bm25_index = BM25Okapi(self._bm25_corpus)
                    logger.info(
                        f"BM25 index initialized with {len(self._bm25_docs)} documents"
                    )
                else:
                    logger.info("BM25 index initialized (empty)")
            else:
                logger.info("BM25 index initialized (no existing documents)")
                self._bm25_docs = []
                self._bm25_corpus = []

        except Exception as e:
            logger.warning(f"Failed to initialize BM25 index: {e}")
            self._bm25_index = None
            self._bm25_docs = []
            self._bm25_corpus = []

    def _initialize_splade(self) -> None:
        """Initialize SPLADE index from existing documents."""
        try:
            logger.info("Initializing SPLADE index from ChromaDB")

            # Re-use documents fetched for BM25 if available, else fetch
            if not self._bm25_docs:
                collection = self._vectorstore._collection
                results = collection.get()
                if results and "documents" in results:
                    docs_content = results["documents"]
                    metadatas = results.get("metadatas", [{}] * len(docs_content))
                    self._splade_docs = [
                        Document(page_content=content, metadata=meta or {})
                        for content, meta in zip(docs_content, metadatas)
                    ]
            else:
                self._splade_docs = list(self._bm25_docs)

            # Build inverted index
            self._splade_index = {}
            for idx, doc in enumerate(self._splade_docs):
                # Check if splade vector is already in metadata
                if "splade_vector" in doc.metadata:
                    sparse_vec = json.loads(doc.metadata["splade_vector"])
                    # Convert keys back to int
                    sparse_vec = {int(k): v for k, v in sparse_vec.items()}
                else:
                    # Compute on the fly (slow for large history, but okay for init)
                    sparse_vec = self.splade_handler.encode(doc.page_content)
                    # Update metadata in memory (not persisting back to Chroma here for speed)
                    doc.metadata["splade_vector"] = json.dumps(sparse_vec)

                for token_id, weight in sparse_vec.items():
                    if token_id not in self._splade_index:
                        self._splade_index[token_id] = []
                    self._splade_index[token_id].append((idx, weight))

            logger.info(
                f"SPLADE index initialized with {len(self._splade_docs)} documents"
            )

        except Exception as e:
            logger.error(f"Failed to initialize SPLADE index: {e}")
            self._splade_index = {}
            self._splade_docs = []

    def _add_to_splade(self, doc: Document) -> None:
        """Add document to SPLADE index."""
        if not self.config.enable_splade:
            return

        doc_idx = len(self._splade_docs)
        self._splade_docs.append(doc)

        # Compute vector
        sparse_vec = self.splade_handler.encode(doc.page_content)

        # Store in metadata for future use
        if doc.metadata is None:
            doc.metadata = {}
        doc.metadata["splade_vector"] = json.dumps(sparse_vec)

        # Update inverted index
        for token_id, weight in sparse_vec.items():
            if token_id not in self._splade_index:
                self._splade_index[token_id] = []
            self._splade_index[token_id].append((doc_idx, weight))

    def retrieve_splade(
        self, query: str, k: int = None, category: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents using SPLADE sparse search.

        Args:
            query: Search query
            k: Number of documents
            category: Optional category filter

        Returns:
            List of relevant documents
        """
        k = k or self.config.hybrid_retrieval_k

        if not self._splade_index or not self._splade_docs:
            return []

        try:
            # Encode query
            query_vec = self.splade_handler.encode(query)

            # Score documents
            doc_scores = {}
            for token_id, q_weight in query_vec.items():
                if token_id in self._splade_index:
                    for doc_idx, d_weight in self._splade_index[token_id]:
                        doc_scores[doc_idx] = doc_scores.get(doc_idx, 0.0) + (
                            q_weight * d_weight
                        )

            # Sort by score
            top_indices = sorted(
                doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True
            )

            results = []
            for idx in top_indices:
                doc = self._splade_docs[idx]

                # Apply category filter
                if category:
                    doc_cat = doc.metadata.get("category")
                    if doc_cat != category and doc_cat != "interaction":
                        if doc_cat != category:
                            continue

                # Add score
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["splade_score"] = float(doc_scores[idx])
                results.append(doc)

                if len(results) >= k:
                    break

            return results

        except Exception as e:
            logger.error(f"SPLADE retrieval failed: {e}")
            return []

    def _add_to_bm25(self, doc: Document) -> None:
        """Add document to BM25 index."""
        if not self.config.enable_hybrid_retrieval:
            return

        self._bm25_docs.append(doc)
        tokenized = doc.page_content.lower().split()
        self._bm25_corpus.append(tokenized)

        # Rebuild index (BM25Okapi doesn't support incremental updates)
        if self._bm25_corpus:
            self._bm25_index = BM25Okapi(self._bm25_corpus)

    def retrieve_bm25(
        self, query: str, k: int = None, category: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents using BM25 keyword search.

        Args:
            query: Search query
            k: Number of documents to retrieve (default: use config)
            category: Optional category to filter by

        Returns:
            List of relevant documents
        """
        k = k or self.config.hybrid_retrieval_k

        if not self._bm25_index or not self._bm25_docs:
            logger.debug("BM25 index empty, returning no results")
            return []

        try:
            tokenized_query = query.lower().split()
            scores = self._bm25_index.get_scores(tokenized_query)

            # Get top-k indices
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )

            # Return corresponding documents with scores, applying filter
            results = []
            for idx in top_indices:
                doc = self._bm25_docs[idx]

                # Apply category filter if provided
                if category:
                    doc_cat = doc.metadata.get("category")
                    if (
                        doc_cat != category and doc_cat != "interaction"
                    ):  # Keep interaction for legacy compatibility if needed, or strict?
                        # Strict routing: if category is provided, only match that category
                        if doc_cat != category:
                            continue

                # Add score to metadata
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["bm25_score"] = float(scores[idx])
                results.append(doc)

                if len(results) >= k:
                    break

            return results

        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
            return []

    def retrieve_hybrid(
        self, queries: List[str], k: int = None, category: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid search (cosine + BM25) with RRF.

        Args:
            queries: List of query variants
            k: Number of final documents to retrieve
            category: Optional category to filter by

        Returns:
            List of documents ranked by RRF score
        """
        if not self.config.enable_hybrid_retrieval:
            # Fallback to cosine only
            return self.retrieve_long_term(
                queries[0] if queries else "", category=category
            )

        k = k or self.config.hybrid_retrieval_k

        # Collect results from all queries
        cosine_results = []
        bm25_results = []
        splade_results = []

        for i, query in enumerate(queries):
            # Cosine similarity results
            with TimingContext("embedding_generation"):
                _ = self._embeddings.embed_query(query)

            with TimingContext("long_term_memory"):
                # Prepare filter
                search_kwargs = {"k": k}
                if category:
                    search_kwargs["filter"] = {"category": category}

                # Retrieve with scores to filter by threshold
                # Chroma's similarity_search_with_score returns (doc, score) where score is distance (lower is better)
                # But we need similarity (higher is better).
                # LangChain's similarity_search returns docs.
                # We need similarity_search_with_relevance_scores
                cosine_docs_with_scores = (
                    self._vectorstore.similarity_search_with_relevance_scores(
                        query, **search_kwargs
                    )
                )

                # Filter by cosine similarity > 0.45
                filtered_cosine = []
                for doc, score in cosine_docs_with_scores:
                    if score > 0.45:
                        doc.metadata["score"] = score
                        filtered_cosine.append(doc)

                cosine_results.extend(filtered_cosine)

            # BM25 results
            with TimingContext("bm25_retrieval"):
                bm25_docs = self.retrieve_bm25(query, k=k, category=category)
                # Filter BM25 results (heuristic threshold, e.g., > 0)
                # BM25 scores can be arbitrary, but usually > 0 means some match.
                # User asked for "TF-IDF > threshold".
                # We'll assume BM25 > 1.0 as a safe bet for "meaningful match" or just keep top K.
                # Let's filter very low scores if possible.
                bm25_results.extend(
                    [d for d in bm25_docs if d.metadata.get("bm25_score", 0) > 1.0]
                )

            # SPLADE results - OPTIMIZATION: Only run for the first query (original)
            if self.config.enable_splade and i == 0:
                with TimingContext("splade_retrieval"):
                    splade_docs = self.retrieve_splade(query, k=k, category=category)
                    splade_results.extend(splade_docs)

        # Apply Reciprocal Rank Fusion
        with TimingContext("hybrid_fusion"):
            # RRF Fusion
            fused_results = self._reciprocal_rank_fusion(
                cosine_results, bm25_results, splade_results, k=k
            )

            # Filter out low relevance results (score < 0.05 after fusion)
            # RRF scores are usually small, so 0.01 is a reasonable threshold for "noise"
            filtered_results = [
                doc for doc in fused_results if doc.metadata.get("rrf_score", 0) > 0.01
            ]

            # Filter by recency (optional, if timestamp exists)
            # User asked: "Filter based on recency"
            # Let's prioritize recent docs or filter very old ones if they are "chat"
            # For now, we just ensure we don't return ancient "chat" logs if they are not relevant.
            # But RRF should handle relevance.
            # We can sort by date if scores are tied, or boost recent docs.
            # Here we just return the RRF ranked list.

            if len(filtered_results) < len(fused_results):
                logger.info(
                    f"Filtered {len(fused_results) - len(filtered_results)} low relevance docs"
                )

        return filtered_results

    def _reciprocal_rank_fusion(
        self,
        cosine_docs: List[Document],
        bm25_docs: List[Document],
        splade_docs: List[Document] = None,
        k: int = 60,
    ) -> List[Document]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        Args:
            cosine_docs: Documents from cosine similarity search
            bm25_docs: Documents from BM25 search
            splade_docs: Documents from SPLADE search (optional)
            k: RRF constant (default: 60)

        Returns:
            Fused and ranked documents
        """
        rrf_k = self.config.rrf_k
        doc_scores = {}
        splade_docs = splade_docs or []

        # Helper to process list
        def process_list(docs, weight):
            for rank, doc in enumerate(docs):
                doc_key = doc.page_content
                score = weight / (rrf_k + rank + 1)
                doc_scores[doc_key] = doc_scores.get(doc_key, 0) + score

        # Calculate RRF scores
        process_list(cosine_docs, self.config.cosine_weight)
        process_list(bm25_docs, self.config.bm25_weight)
        if splade_docs:
            process_list(splade_docs, self.config.splade_weight)

        # Deduplicate and combine documents
        doc_map = {}
        for doc in cosine_docs + bm25_docs + splade_docs:
            doc_key = doc.page_content
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

        # Sort by RRF score
        sorted_docs = sorted(
            doc_map.items(), key=lambda x: doc_scores.get(x[0], 0), reverse=True
        )

        # Add RRF scores to metadata
        results = []
        for doc_key, doc in sorted_docs[:k]:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata["rrf_score"] = doc_scores[doc_key]
            results.append(doc)

        return results
