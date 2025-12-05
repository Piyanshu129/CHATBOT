#!/usr/bin/env python3
"""Main entry point for the chatbot application."""

import sys
from pathlib import Path

# Add parent directory to Python path if needed
bot_dir = Path(__file__).resolve().parent
parent_dir = bot_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from bot_custom.interface.cli import main

if __name__ == "__main__":
    sys.exit(main())


# For ollama
# python main.py --backend ollama


# For huggingface
# python main.py --backend huggingface


# For vllm
# python --backend vllm --vllm-url http://172.17.1.229:2525/generate
