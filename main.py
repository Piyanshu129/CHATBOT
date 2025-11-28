#!/usr/bin/env python3
"""Main entry point for the chatbot application."""

import sys
from pathlib import Path

# Add parent directory to Python path if needed
bot_dir = Path(__file__).resolve().parent
parent_dir = bot_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from bot.interface.cli import main

if __name__ == "__main__":
    sys.exit(main())
