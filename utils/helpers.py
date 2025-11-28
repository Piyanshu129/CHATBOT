"""Helper functions and utilities."""

import logging
import sys
import warnings
from typing import Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set specific loggers to WARNING to reduce noise
    logging.getLogger('chromadb').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('langchain').setLevel(logging.WARNING)


def filter_special_tokens(text: str) -> str:
    """
    Filter out special tokens from model output.
    
    Args:
        text: Text that may contain special tokens
        
    Returns:
        Cleaned text with special tokens removed
    """
    # Common special tokens to filter
    special_tokens = [
        "<|eot_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|begin_of_text|>",
    ]
    
    cleaned_text = text
    for token in special_tokens:
        cleaned_text = cleaned_text.replace(token, "")
    
    return cleaned_text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length of text
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix
