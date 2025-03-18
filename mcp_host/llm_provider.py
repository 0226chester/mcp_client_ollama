import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union

from anthropic import Anthropic

from .chat_session import Message, ContentBlock
from .config import LLMProviderConfig

logger = logging.getLogger("mcp-host")

# Constants
MAX_RETRIES = 5
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 30     # seconds

class LLMProvider:
    """Communicate with the Anthropic Claude API."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20240620"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables or .env file")
        
        self.client = Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized LLM provider with model: {model}")
    
    async def create_message(
        self, 
        messages: List[Message], 
        tools: List[Dict[str, Any]], 
        prompt: str = None
    ) -> Message:
        """Create a message using the LLM."""
        # Convert our messages to Anthropic's format
        anthropic_messages = []
        
        for msg in messages:
            content = []
            
            # Process based on message role
            if msg.role in ["user", "assistant"]:
                # Add text content if present
                text_content = msg.get_text_content()
                if text_content:
                    content.append({"type": "text", "text": text_content})
                
                # Add tool calls if present (only for assistant messages)
                if msg.role == "assistant":
                    for tool_id, tool_name, tool_input in msg.get_tool_calls():
                        content.append({
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": tool_input
                        })
                
                # Only add message if it has content
                if content:
                    anthropic_messages.append({"role": msg.role, "content": content})
                    
            elif msg.role == "tool":
                # This is a tool response, handled differently
                for block in msg.content:
                    if block.type == "tool_result":
                        anthropic_messages.append({
                            "role": "tool",  # Special "tool" role
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id,
                                "content": block.content
                            }]
                        })
        
        # Add the new prompt if provided
        if prompt:
            anthropic_messages.append({
                "role": "user", 
                "content": [{"type": "text", "text": prompt}]
            })
        
        logger.debug(f"Sending {len(anthropic_messages)} messages to Claude")
        
        # Log message roles for debugging
        roles = [m.get("role") for m in anthropic_messages]
        logger.debug(f"Message roles sequence: {roles}")
        
        # Convert tools to Anthropic's format
        anthropic_tools = []
        for tool in tools:
            anthropic_tools.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool["inputSchema"]
            })
        
        # Make API call with retries
        retries = 0
        backoff = INITIAL_BACKOFF
        while True:
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    messages=anthropic_messages,
                    tools=anthropic_tools if tools else None
                )
                
                # Process the response
                content_blocks = []
                for content in response.content:
                    if content.type == "text":
                        content_blocks.append(ContentBlock(
                            type="text",
                            text=content.text
                        ))
                    elif content.type == "tool_use":
                        content_blocks.append(ContentBlock(
                            type="tool_use",
                            id=content.id,
                            name=content.name,
                            input=content.input
                        ))
                
                return Message(role="assistant", content=content_blocks)
            
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error from LLM API: {error_msg}")
                
                # Check if it's an overloaded error
                if "overloaded_error" in error_msg.lower() and retries < MAX_RETRIES:
                    retries += 1
                    logger.warning(f"Claude is overloaded, backing off (attempt {retries}/{MAX_RETRIES}, {backoff}s)")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF)
                    continue
                else:
                    raise


class LLMProviderFactory:
    """Factory for creating LLM providers based on configuration."""
    
    @staticmethod
    def create_provider(config: Optional[LLMProviderConfig] = None) -> Any:
        """Create an LLM provider based on configuration."""
        if not config:
            # Default to Anthropic if no config provided
            return LLMProvider()
        
        provider_type = config.type.lower()
        
        if provider_type == "anthropic":
            return LLMProvider(model=config.model)
        elif provider_type == "ollama":
            # Import OllamaProvider only when needed to avoid circular imports
            from .ollama_provider import OllamaProvider
            return OllamaProvider(
                model=config.model,
                url=config.url or "http://localhost:11434",
                parameters=config.parameters
            )
        else:
            logger.error(f"Unknown LLM provider type: {config.type}")
            # Fall back to Anthropic as default
            return LLMProvider()