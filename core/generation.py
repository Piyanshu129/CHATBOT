"""Chain of Verification (CoV) generation module."""

import logging

from langchain_core.language_models import BaseLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import BotConfig
from ..utils.timing import TimingContext

logger = logging.getLogger(__name__)


class ChainOfVerification:
    """
    Implements Chain of Verification (CoV) to reduce hallucinations.
    Steps:
    1. Generate Baseline Response
    2. Plan Verification Questions
    3. Execute Verification (Answer Questions)
    4. Generate Final Verified Response
    """

    BASELINE_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Answer the user's question using the provided context.
Context:
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    VERIFICATION_PLAN_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a fact-checker. Given a question and a baseline response, generate 3-5 verification questions to check the factual accuracy of the response.
Focus on facts, numbers, dates, and specific entities.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Original Question: {question}
Baseline Response: {response}

Verification Questions:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    VERIFICATION_EXECUTION_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Answer the verification question using the provided context. Be concise.

Context:
{context}

<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    FINAL_RESPONSE_PROMPT = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant. Generate a final, verified response to the user's question.
Use the Baseline Response and the Verification Results to ensure accuracy.
If the Baseline Response was correct, you can reuse it. If it had errors, correct them based on the Verification Results.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Original Question: {question}

Baseline Response:
{baseline_response}

Verification Results:
{verification_results}

Final Verified Response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def __init__(self, config: BotConfig, llm: BaseLLM):
        """
        Initialize Chain of Verification.

        Args:
            config: Bot configuration
            llm: Language model
        """
        self.config = config
        self.llm = llm

        self.baseline_chain = (
            ChatPromptTemplate.from_template(self.BASELINE_PROMPT)
            | llm
            | StrOutputParser()
        )
        self.plan_chain = (
            ChatPromptTemplate.from_template(self.VERIFICATION_PLAN_PROMPT)
            | llm
            | StrOutputParser()
        )
        self.execute_chain = (
            ChatPromptTemplate.from_template(self.VERIFICATION_EXECUTION_PROMPT)
            | llm
            | StrOutputParser()
        )
        self.final_chain = (
            ChatPromptTemplate.from_template(self.FINAL_RESPONSE_PROMPT)
            | llm
            | StrOutputParser()
        )

        logger.info("Initialized ChainOfVerification")

    def should_verify(
        self,
        question: str,
        context: str,
        baseline_response: str,
        category: str = None,
        retrieval_score: float = 1.0,
    ) -> bool:
        """
        Determine if verification is needed.
        Skip if:
        - Response is short (< 20 words)
        - Context is empty
        - Question is simple chat
        - Category is CHAT
        - Retrieval score is low (< 0.5)
        """
        # Rule 1: Short response
        if len(baseline_response.split()) < 20:
            logger.info("Skipping CoV: Response too short")
            return False

        # Rule 2: Empty context
        if not context or len(context.strip()) < 10:
            logger.info("Skipping CoV: No context to verify against")
            return False

        # Rule 3: Simple chat keywords in question
        chat_keywords = {"hi", "hello", "thanks", "bye", "cool", "ok"}
        if (
            any(w in question.lower().split() for w in chat_keywords)
            and len(question.split()) < 5
        ):
            logger.info("Skipping CoV: Simple chat detected")
            return False

        # Rule 4: Category is CHAT
        if category == "chat":
            logger.info("Skipping CoV: Category is CHAT")
            return False

        # Rule 5: Low retrieval score
        if retrieval_score < 0.5:
            logger.info(f"Skipping CoV: Low retrieval score ({retrieval_score:.2f})")
            return False

        return True

    def generate(
        self,
        question: str,
        context: str,
        category: str = None,
        retrieval_score: float = 1.0,
    ) -> str:
        """
        Generate a verified response.

        Args:
            question: User question
            context: Retrieved context
            category: Query category
            retrieval_score: Max retrieval score
        """
        with TimingContext("cov_generation"):
            # 1. Baseline Response
            logger.debug("Generating baseline response...")
            baseline_response = self.baseline_chain.invoke(
                {"question": question, "context": context}
            )

            # Check if we should verify
            if not self.should_verify(
                question, context, baseline_response, category, retrieval_score
            ):
                return baseline_response

            # 2. Plan Verification
            logger.debug("Planning verification...")
            plan = self.plan_chain.invoke(
                {"question": question, "response": baseline_response}
            )
            verification_questions = [q.strip() for q in plan.split("\n") if "?" in q]

            if not verification_questions:
                logger.debug("No verification questions generated, returning baseline.")
                return baseline_response

            # 3. Execute Verification
            logger.debug(
                f"Executing {len(verification_questions)} verification questions..."
            )
            verification_results = []
            for vq in verification_questions[:3]:  # Limit to 3 to save time
                answer = self.execute_chain.invoke({"question": vq, "context": context})
                verification_results.append(f"Q: {vq}\nA: {answer}")

            verification_text = "\n\n".join(verification_results)

            # 4. Final Response
            logger.debug("Generating final verified response...")
            final_response = self.final_chain.invoke(
                {
                    "question": question,
                    "baseline_response": baseline_response,
                    "verification_results": verification_text,
                }
            )

            return final_response
