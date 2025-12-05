"""Memory encoder for structured memory extraction."""

import logging
from typing import List, Dict
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import BotConfig
from ..utils.timing import TimingContext

logger = logging.getLogger(__name__)


class MemoryEncoder:
    """
    Encodes conversation interactions into structured memory categories:
    - Episodic: Events, specific interactions
    - Semantic: Facts, knowledge
    - User Profile: Preferences, personal details
    """

    EXTRACTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a memory expert. Analyze the following conversation and extract new information into three categories:
1. User Profile: Facts about the user (name, preferences, job, etc.)
2. Semantic Memory: General facts or knowledge discussed
3. Episodic Memory: Summary of the specific interaction event

Format:
Category: [Profile/Semantic/Episodic]
Content: [The extracted fact]

Return only the extracted facts. If nothing new, return "None".

<|eot_id|><|start_header_id|>user<|end_header_id|>
User: {user_input}
Assistant: {bot_response}

Extract memory:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def __init__(self, config: BotConfig, llm: BaseLLM):
        """
        Initialize memory encoder.

        Args:
            config: Bot configuration
            llm: Language model
        """
        self.config = config
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_template(self.EXTRACTION_PROMPT)
        self.chain = self.prompt | llm | StrOutputParser()

        logger.info("Initialized MemoryEncoder")

    def encode_interaction(
        self, user_input: str, bot_response: str
    ) -> List[Dict[str, str]]:
        """
        Extract structured memories from an interaction.

        Args:
            user_input: User message
            bot_response: Bot response

        Returns:
            List of dicts with 'category' and 'content'
        """
        memories = []

        try:
            with TimingContext("memory_encoding"):
                response = self.chain.invoke(
                    {"user_input": user_input, "bot_response": bot_response}
                )

                lines = response.strip().split("\n")
                current_category = None

                for line in lines:
                    line = line.strip()
                    if not line or line == "None":
                        continue

                    if line.startswith("Category:"):
                        cat_str = line.split(":", 1)[1].strip().lower()
                        if "profile" in cat_str:
                            current_category = "user_profile"
                        elif "semantic" in cat_str:
                            current_category = "semantic"
                        elif "episodic" in cat_str:
                            current_category = "episodic"
                        else:
                            current_category = "interaction"  # Default

                    elif line.startswith("Content:") and current_category:
                        content = line.split(":", 1)[1].strip()
                        memories.append(
                            {"category": current_category, "content": content}
                        )

        except Exception as e:
            logger.error(f"Memory encoding failed: {e}")

        return memories
