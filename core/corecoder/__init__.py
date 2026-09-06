"""CoreCoder - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.4.0"

from core.corecoder.agent import Agent
from core.corecoder.llm import LLM
from core.corecoder.config import Config
from core.corecoder.tools import ALL_TOOLS

__all__ = ["Agent", "LLM", "Config", "ALL_TOOLS", "__version__"]
