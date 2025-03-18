import os
import json
import asyncio
import logging
import uuid
import aiohttp
from typing import List, Dict, Any, Optional, Union

from .chat_session import Message, ContentBlock

logger = logging.getLogger("mcp-host")

# Constants
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 30     # seconds

class OllamaProvider:
    """Communicate with the Ollama API."""
    
    def __init__(
        self, 
        model: str = "llama3", 
        url: str = "http://localhost:11434", 
        parameters: Dict[str, Any] = None
    ):
        self.model = model
        self.parameters = parameters or {}
        self.url = url.rstrip('/')
        logger.info(f"Initialized Ollama provider with model: {model} at {url}")
        
        # Set default parameters if not provided
        if "temperature" not in self.parameters:
            self.parameters["temperature"] = 0.7
        if "num_predict" not in self.parameters:
            self.parameters["num_predict"] = 1024
    
    async def connect(self) -> bool:
        """Establish connection to Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                # Check models endpoint as a simple ping
                async with session.get(f"{self.url}/api/tags") as response:
                    if response.status != 200:
                        logger.error(f"Failed to connect to Ollama API: {response.status}")
                        return False
                    
                    # Check if our model is available
                    data = await response.json()
                    models = [model.get("name") for model in data.get("models", [])]
                    
                    if not models:
                        logger.warning("No models found in Ollama")
                    elif self.model not in models:
                        logger.warning(f"Model {self.model} not found in Ollama. " +
                                     f"Available models: {', '.join(models)}")
                    
                    logger.info(f"Successfully connected to Ollama API")
                    return True
        except Exception as e:
            logger.error(f"Error connecting to Ollama API: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from Ollama."""
        # Nothing to do for basic Ollama implementation
        pass
    
    async def create_message(
        self, 
        messages: List[Message], 
        tools: List[Dict[str, Any]] = None, 
        prompt: str = None
    ) -> Message:
        """Create a message using Ollama LLM."""
        # Format messages for Ollama
        ollama_messages = []
        
        # Track if we have any tool results to prepare context
        has_tool_results = False
        
        # Process all messages for Ollama
        for msg in messages:
            # Skip empty messages
            if not msg.content:
                continue
                
            if msg.role == "user" or msg.role == "assistant":
                # Get text content from the message
                text_content = msg.get_text_content()
                
                if text_content:
                    ollama_messages.append({
                        "role": msg.role,
                        "content": text_content
                    })
                    
            elif msg.role == "tool":
                has_tool_results = True
                # For tool responses, add as a tool role in Ollama
                for block in msg.content:
                    if block.type == "tool_result" and block.content:
                        # Get tool content as text
                        tool_content = ""
                        if isinstance(block.content, list):
                            for content_item in block.content:
                                if isinstance(content_item, dict) and content_item.get("type") == "text":
                                    tool_content += content_item.get("text", "")
                        elif isinstance(block.content, str):
                            tool_content = block.content
                            
                        # Find the corresponding tool_call to get the name
                        tool_name = self._find_tool_name_for_response(messages, block.tool_use_id)
                        
                        # Add as tool message
                        ollama_messages.append({
                            "role": "tool", 
                            "content": tool_content,
                            "name": tool_name or "unknown_tool"
                        })
                        
        # Add the new prompt if provided
        if prompt:
            ollama_messages.append({
                "role": "user",
                "content": prompt
            })
            
        # Avoid empty message list
        if not ollama_messages:
            ollama_messages.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
        
        # Prepare request payload for chat
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False
        }
        
        # Add additional parameters from config
        if self.parameters:
            payload["options"] = self.parameters
        
        # Add tools if provided and supported
        formatted_tools = []
        if tools and self._supports_function_calling():
            for tool in tools:
                # Extract the actual tool name (removing server prefix if present)
                tool_name = tool["name"]
                original_name = tool_name
                
                if "__" in tool_name:
                    _, original_name = tool_name.split("__", 1)
                
                # Format tool according to Ollama's expected structure
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,  # Use full namespaced name 
                        "description": tool.get("description", ""),
                        "parameters": tool["inputSchema"]
                    }
                })
            
            # Add tools to the payload
            if formatted_tools:
                payload["tools"] = formatted_tools
                logger.debug(f"Sending {len(formatted_tools)} tools to Ollama")
        
        # Log what we're sending to Ollama
        logger.debug(f"Sending {len(ollama_messages)} messages to Ollama, has_tool_results={has_tool_results}")
        
        # Make API call with retries
        retries = 0
        backoff = INITIAL_BACKOFF
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    # Ollama API endpoint for chat completion
                    url = f"{self.url}/api/chat"
                    
                    logger.debug(f"Sending request to {url}")
                    
                    async with session.post(url, json=payload, timeout=60) as response:
                        if response.status >= 400:
                            error_text = await response.text()
                            raise RuntimeError(f"Ollama API error ({response.status}): {error_text}")
                            
                        result = await response.json()
                        
                        # Process the response
                        content_blocks = []
                        
                        # Add text response if present
                        if "message" in result and "content" in result["message"] and result["message"]["content"]:
                            content_blocks.append(ContentBlock(
                                type="text",
                                text=result["message"]["content"]
                            ))
                        
                        # Handle tool calls if present
                        if "message" in result and "tool_calls" in result["message"]:
                            tool_calls = result["message"].get("tool_calls", [])
                            logger.debug(f"Received {len(tool_calls)} tool calls from Ollama")
                            
                            for tool_call in tool_calls:
                                if "function" in tool_call:
                                    function_call = tool_call["function"]
                                    tool_name = function_call["name"]
                                    
                                    logger.debug(f"Ollama returned tool call for: {tool_name}")
                                    
                                    # Create tool_use block
                                    content_blocks.append(ContentBlock(
                                        type="tool_use",
                                        id=str(uuid.uuid4()),  # Generate a unique ID
                                        name=tool_name,
                                        input=function_call.get("arguments", {})
                                    ))
                        
                        return Message(role="assistant", content=content_blocks)
                
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error from Ollama API: {error_msg}")
                    
                    # Retry on certain error conditions
                    if (retries < MAX_RETRIES and 
                        ("overloaded" in error_msg.lower() or 
                         "timeout" in error_msg.lower() or
                         "connection" in error_msg.lower())):
                        retries += 1
                        logger.warning(f"Ollama request failed, backing off (attempt {retries}/{MAX_RETRIES}, {backoff}s)")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, MAX_BACKOFF)
                        continue
                    else:
                        # Create an error message
                        return Message(
                            role="assistant",
                            content=[ContentBlock(
                                type="text",
                                text=f"Error communicating with Ollama: {error_msg}"
                            )]
                        )
    
    def _supports_function_calling(self) -> bool:
        """Determine if the current model supports function calling."""
        # Models known to have function calling capability
        function_capable_models = [
            "llama3", "mistral", "mixtral", "gemma", "phi3",
            "neural-chat", "openchat", "wizard", "qwen"
        ]
        
        model_lower = self.model.lower()
        for model_prefix in function_capable_models:
            if model_prefix.lower() in model_lower:
                return True
        
        return False
        
    def _find_tool_name_for_response(self, messages: List[Message], tool_use_id: str) -> Optional[str]:
        """Find the tool name that corresponds to a tool response ID."""
        for msg in messages:
            if msg.role == "assistant":
                for block in msg.content:
                    if block.type == "tool_use" and block.id == tool_use_id:
                        return block.name
        return None