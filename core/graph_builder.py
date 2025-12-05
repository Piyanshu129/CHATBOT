"""Context graph builder for GraphRAG."""

import logging
from typing import List, Dict, Set, Any
from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import BotConfig
from ..utils.timing import TimingContext

logger = logging.getLogger(__name__)


class ContextGraphBuilder:
    """
    Builds a lightweight knowledge graph from retrieved documents
    to identify connections and improve context understanding.
    """

    ENTITY_EXTRACTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a knowledge graph expert. Extract key entities and relationships from the text below.
Format: Entity1 - Relationship - Entity2
Example: 
Python - is a - programming language
ChromaDB - stores - vector embeddings

Return ONLY the relationships, one per line. Limit to top 3 most important relationships.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Text: {text}

Extract relationships:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def __init__(self, config: BotConfig, llm: BaseLLM):
        """
        Initialize graph builder.

        Args:
            config: Bot configuration
            llm: Language model for extraction
        """
        self.config = config
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template(self.ENTITY_EXTRACTION_PROMPT)
        self.chain = self.prompt | llm | StrOutputParser()

        logger.info("Initialized ContextGraphBuilder")

    def build_graph(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Build a graph from documents.

        Args:
            documents: List of retrieved documents

        Returns:
            Dictionary representing the graph (nodes and edges)
        """
        if not documents:
            return {"nodes": [], "edges": []}

        nodes: Set[str] = set()
        edges: List[str] = []

        with TimingContext("graph_building"):
            # For efficiency, we might only process the top K docs or summarize them first
            # Here we process each doc independently
            for doc in documents[:2]:  # Limit to top 2 for speed
                try:
                    relationships = self._extract_relationships(doc.page_content)
                    edges.extend(relationships)

                    # Parse nodes from edges
                    for rel in relationships:
                        parts = rel.split("-")
                        if len(parts) >= 2:
                            nodes.add(parts[0].strip())
                            nodes.add(parts[-1].strip())

                except Exception as e:
                    logger.warning(f"Graph extraction failed for doc: {e}")

        # Enforce strict limit of 3 relations max to prevent context pollution
        edges = edges[:3]

        graph_data = {
            "nodes": list(nodes),
            "edges": edges,
            "summary": self._summarize_graph(edges),
        }

        logger.debug(f"Built graph with {len(nodes)} nodes and {len(edges)} edges")
        return graph_data

    def _extract_relationships(self, text: str) -> List[str]:
        """Extract relationships from text using LLM."""
        response = self.chain.invoke({"text": text[:1000]})  # Truncate
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return lines

    def _summarize_graph(self, edges: List[str]) -> str:
        """Create a text summary of the graph connections."""
        if not edges:
            return ""
        return "Key Connections:\n" + "\n".join(edges[:10])
