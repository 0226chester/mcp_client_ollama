import os
import json
import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("mcp-host")

@dataclass
class ServerConfig:
    """Configuration for an MCP server."""
    type: str = "stdio"  # Default to stdio for backward compatibility
    command: Optional[str] = None  # Required for stdio, not for SSE
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None  # Required for SSE, not for stdio

@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""
    type: str  # "anthropic" or "ollama"
    model: str
    api_key: Optional[str] = None  # For providers that need API keys
    url: Optional[str] = None      # Base URL for the API, mainly for Ollama
    parameters: Dict[str, Any] = field(default_factory=dict)  # Additional parameters

class ConfigLoader:
    """Load and manage configuration."""
    
    @staticmethod
    def load_config(config_path: str = None) -> Tuple[Dict[str, ServerConfig], Optional[LLMProviderConfig]]:
        """Load server and LLM provider configuration from JSON file."""
        if not config_path:
            config_path = os.path.join(os.getcwd(), "config.json")
        
        # Create default config if it doesn't exist
        if not os.path.exists(config_path):
            default_config = {
                "mcpServers": {},
                "llmProvider": {
                    "type": "anthropic",
                    "model": "claude-3-5-sonnet-20240620"
                }
            }
            with open(config_path, "w") as f:
                json.dump(default_config, f, indent=2)
            logger.info(f"Created default config file at {config_path}")
            return {}, None
        
        # Load existing config
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            # Load server configs
            servers = {}
            for name, server_config in config_data.get("mcpServers", {}).items():
                transport_type = server_config.get("type", "stdio")
                
                # Validate configuration based on transport type
                if transport_type == "stdio":
                    if "command" not in server_config:
                        logger.error(f"Server '{name}' is missing required 'command' for stdio transport")
                        continue
                elif transport_type == "sse":
                    if "url" not in server_config:
                        logger.error(f"Server '{name}' is missing required 'url' for SSE transport")
                        continue
                    
                    url = server_config["url"]
                    if not re.match(r'^https?://', url):
                        logger.error(f"Server '{name}' has invalid URL: {url}. Must start with http:// or https://")
                        continue
                else:
                    logger.error(f"Server '{name}' has unsupported transport type: {transport_type}")
                    continue
                
                servers[name] = ServerConfig(
                    type=transport_type,
                    command=server_config.get("command"),
                    args=server_config.get("args", []),
                    env=server_config.get("env", {}),
                    url=server_config.get("url")
                )
            
            # Load LLM provider config
            llm_config = None
            if "llmProvider" in config_data:
                provider_config = config_data["llmProvider"]
                llm_config = LLMProviderConfig(
                    type=provider_config.get("type", "anthropic"),
                    model=provider_config.get("model", "claude-3-5-sonnet-20240620"),
                    api_key=provider_config.get("api_key"),
                    url=provider_config.get("url"),
                    parameters=provider_config.get("parameters", {})
                )
            
            return servers, llm_config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            return {}, None
