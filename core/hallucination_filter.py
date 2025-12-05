"""Hallucination filter and document relevance checker (Self-RAG)."""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import BotConfig
from ..utils.timing import TimingContext

logger = logging.getLogger(__name__)


class HallucinationFilter:
    """
    Filters retrieved documents based on relevance to the query.
    Implements a simplified Self-RAG approach.
    """

    RELEVANCE_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a relevance grader. You will be given a user question and a retrieved document.
Your task is to determine if the document contains information relevant to answering the question.

Return only "yes" or "no".

<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}

Document:
{document}

Is this document relevant to the question? (yes/no)<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def __init__(self, config: BotConfig, llm: BaseLLM):
        """
        Initialize hallucination filter.

        Args:
            config: Bot configuration
            llm: Language model for verification
        """
        self.config = config
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template(self.RELEVANCE_PROMPT)
        self.chain = self.prompt | llm | StrOutputParser()

        logger.info("Initialized HallucinationFilter")

    def filter_documents(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Filter documents by relevance.

        Args:
            query: User query
            documents: List of retrieved documents

        Returns:
            List of relevant documents
        """
        if not documents:
            return []

        relevant_docs = []

        # We can run this in parallel if needed, but for now sequential
        # to avoid overwhelming the LLM or hitting rate limits
        with TimingContext("hallucination_filter"):
            for doc in documents:
                if self.check_relevance(query, doc):
                    relevant_docs.append(doc)
                else:
                    logger.debug(
                        f"Filtered out irrelevant document: {doc.page_content[:50]}..."
                    )

        logger.info(
            f"Filtered {len(documents)} docs to {len(relevant_docs)} relevant docs"
        )
        return relevant_docs

    def check_relevance(self, query: str, doc: Document) -> bool:
        """
        Check if a single document is relevant.

        Args:
            query: User query
            doc: Document to check

        Returns:
            True if relevant
        """
        try:
            # Quick check: if score is very high (from reranker), skip LLM check?
            # For now, we trust the LLM grader.

            response = self.chain.invoke(
                {
                    "question": query,
                    "document": doc.page_content[:1000],  # Truncate for speed
                }
            )

            result = response.strip().lower()
            is_relevant = "yes" in result

            if is_relevant:
                # Add relevance score to metadata
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["relevance_check"] = "pass"

            return is_relevant

        except Exception as e:
            logger.warning(f"Relevance check failed: {e}, assuming relevant")
            return True
