"""LLM model initialization and factory."""

import torch
import logging
from typing import Union
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from ..config import BotConfig
from ..utils.timing import TimingContext

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory class for creating LLM instances."""

    @staticmethod
    def create_llm(config: BotConfig) -> Union[ChatOllama, HuggingFacePipeline]:
        """
        Create an LLM instance based on the configuration.

        Args:
            config: Bot configuration object

        Returns:
            LLM instance (ChatOllama or HuggingFacePipeline)

        Raises:
            ValueError: If backend is not supported
            RuntimeError: If model initialization fails
        """
        if config.llm_backend == "ollama":
            with TimingContext("model_loading"):
                return LLMFactory._create_ollama_llm(config)
        elif config.llm_backend == "huggingface":
            with TimingContext("model_loading"):
                return LLMFactory._create_huggingface_llm(config)
        elif config.llm_backend == "vllm":
            with TimingContext("model_loading"):
                return LLMFactory._create_vllm_llm(config)
        else:
            raise ValueError(f"Unsupported LLM backend: {config.llm_backend}")

    @staticmethod
    def _create_ollama_llm(config: BotConfig) -> ChatOllama:
        """
        Create an Ollama LLM instance.

        Args:
            config: Bot configuration object

        Returns:
            ChatOllama instance
        """
        try:
            logger.info(f"Initializing Ollama model: {config.ollama_model}")
            llm = ChatOllama(
                model=config.ollama_model,
                temperature=config.temperature,
            )
            logger.info("Ollama model initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Ollama model: {e}")
            raise RuntimeError(f"Ollama initialization failed: {e}") from e

    @staticmethod
    def _create_vllm_llm(config: BotConfig) -> ChatOpenAI:
        """
        Create a vLLM instance using OpenAI-compatible API.

        Args:
            config: Bot configuration object

        Returns:
            ChatOpenAI instance pointing to vLLM server
        """
        try:
            logger.info(
                f"Initializing vLLM model: {config.vllm_model} at {config.vllm_url}"
            )
            llm = ChatOpenAI(
                model=config.vllm_model,
                openai_api_key="EMPTY",  # vLLM usually doesn't require a key
                openai_api_base=config.vllm_url,
                temperature=config.temperature,
                max_tokens=config.max_new_tokens,
                top_p=config.top_p,
            )
            # Verify connection by doing a dry run or just logging
            logger.info("vLLM client initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize vLLM client: {e}")
            raise RuntimeError(f"vLLM initialization failed: {e}") from e

    @torch.inference_mode()
    def _create_huggingface_llm(config: BotConfig) -> HuggingFacePipeline:
        """
        Create a HuggingFace Pipeline LLM instance.

        Args:
            config: Bot configuration object

        Returns:
            HuggingFacePipeline instance

        Note:
            This requires significant GPU memory. Consider using Ollama for
            resource-constrained environments.
        """
        try:
            logger.info(f"Loading HuggingFace model: {config.huggingface_model}")
            logger.info(
                "This may take a while and requires significant RAM/GPU memory..."
            )

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.huggingface_model)

            # Determine torch dtype
            torch_dtype = (
                torch.float16 if config.torch_dtype == "float16" else torch.float32
            )

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                config.huggingface_model,
                torch_dtype=torch_dtype,
                device_map=config.device_map,
                trust_remote_code=True,
            )

            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                temperature=config.temperature,
                top_p=config.top_p,
                repetition_penalty=config.repetition_penalty,
                return_full_text=False,
            )

            llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("HuggingFace model initialized successfully")
            return llm

        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {e}")
            raise RuntimeError(f"HuggingFace initialization failed: {e}") from e
