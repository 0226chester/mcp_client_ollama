import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union

from .chat_session import Message, ContentBlock
from .config import LLMProviderConfig

logger = logging.getLogger("mcp-host")

# Constants
MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 30     # seconds


class LLMProviderFactory:
    """Factory for creating LLM providers based on configuration."""    
    @staticmethod
    def create_provider(config: Optional[LLMProviderConfig] = None) -> Any:
        """Create an LLM provider based on configuration."""
            # Import OllamaProvider only when needed to avoid circular imports
        from .ollama_provider import OllamaProvider
        return OllamaProvider(
            model=config.model if config and config.model else "llama3",
            url=config.url or "http://localhost:11434",
            parameters=config.parameters or {}
        )
