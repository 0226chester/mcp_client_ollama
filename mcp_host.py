import os
import json
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure current directory for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("mcp-host")

# Try to import dotenv for environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("dotenv not installed. Environment variables will not be loaded from .env file.")

# Import MCP Host modules
from mcp_host.config import ConfigLoader, ServerConfig, LLMProviderConfig
from mcp_host.client_manager import MCPClientManager
from mcp_host.ollama_provider import OllamaProvider
from mcp_host.chat_session import ChatSession, Message, ContentBlock

# Constants
DEFAULT_MESSAGE_WINDOW = 10
MAX_RETRIES = 3
INITIAL_BACKOFF = 1  # seconds
MAX_BACKOFF = 30     # seconds

async def run_chat_session(
    config_path: str = None, 
    model: str = None, 
    message_window: int = DEFAULT_MESSAGE_WINDOW,
    provider_overrides: Dict[str, Any] = None
):
    """Run the main chat session."""
    mcp_manager = None
    
    try:
        # Load server configurations
        server_configs, llm_config = ConfigLoader.load_config(config_path)
        
        if not server_configs:
            logger.warning("No MCP servers configured.")
            print("No MCP servers configured. Please add servers to your config file.")
            return
        
        # Create MCP client manager
        mcp_manager = MCPClientManager(server_configs)
        await mcp_manager.initialize_clients()
        
        if not mcp_manager.clients:
            logger.error("Failed to initialize any MCP clients.")
            print("Failed to initialize any MCP clients. Check your configuration and logs.")
            return
        
        # Prepare LLM config
        if not llm_config:
            llm_config = LLMProviderConfig(
                type="ollama",
                model=model or "llama3",  # Default to llama3 if not specified
                url="http://localhost:11434",  # Default Ollama URL
                parameters={}
            )
            
        # Apply provider overrides if specified
        if provider_overrides:
            # Override existing config with command line parameters
            if "type" in provider_overrides:
                llm_config.type = provider_overrides["type"]
            if "url" in provider_overrides:
                llm_config.url = provider_overrides["url"]
            if "model" in provider_overrides or model:
                llm_config.model = model or provider_overrides.get("model")
        
        # Override model if specified in command line
        elif model and llm_config:
            llm_config.model = model
                   
        # Create Ollama provider
        llm_provider = OllamaProvider(
            model=llm_config.model,
            url=llm_config.url or "http://localhost:11434",
            parameters=llm_config.parameters or {}
        )
        
        # Create chat session
        chat_session = ChatSession(
            llm_provider=llm_provider,
            mcp_manager=mcp_manager,
            message_window=message_window
        )
        
        # Print provider info
        provider_type = "ollama"
        provider_model = llm_config.model
        provider_url = llm_config.url or "http://localhost:11434"
        
        print("\nMCP Host initialized.")
        print(f"Provider: {provider_type}")
        print(f"Model: {provider_model}")
        print(f"Ollama URL: {provider_url}")
        print(f"Connected MCP Servers: {len(mcp_manager.clients)}")
        
        print("\nType 'exit' or 'quit' to quit.")
        print("Type 'tools' to list available tools.")
        print("Type 'servers' to list connected servers.\n")
        
        # Main chat loop
        while True:
            try:
                prompt = input("\nYou: ").strip()
                
                if prompt.lower() in ["exit", "quit"]:
                    print("\nExiting...")
                    break
                
                if prompt.lower() == "tools":
                    # List available tools
                    tools = await mcp_manager.get_all_tools()
                    print("\nAvailable Tools:")
                    for tool in tools:
                        print(f"  - {tool['name']}: {tool.get('description', 'No description')}")
                    continue
                
                if prompt.lower() == "servers":
                    # List connected servers
                    print("\nConnected Servers:")
                    for name, client in mcp_manager.clients.items():
                        print(f"  - {name}: {client.config.type}")
                    continue
                
                print("\nAssistant: ", end="", flush=True)
                response = await chat_session.process_prompt(prompt)
                print(f"{response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error in chat session: {str(e)}")
                print(f"\nError: {str(e)}")
    
    finally:
        # Clean up resources
        if mcp_manager is not None:
            logger.debug("Starting cleanup of MCP manager")
            await mcp_manager.shutdown_all()
            logger.debug("MCP manager cleanup complete")

def update_config_with_provider_info(
    config_path: str, 
    provider_type: str, 
    model: str = None, 
    url: str = None,
    parameters: Dict[str, Any] = None
) -> bool:
    """Update the config file with provider information."""
    try:
        # Default config path if not provided
        if not config_path:
            config_path = os.path.join(os.getcwd(), "config.json")
            
        # Load existing config or create new one
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            config = {"mcpServers": {}}
        
        # Create or update llmProvider section
        if "llmProvider" not in config:
            config["llmProvider"] = {}
            
        # Set provider type
        config["llmProvider"]["type"] = provider_type
        
        # Set model if provided
        if model:
            config["llmProvider"]["model"] = model
            
        # Set URL if provided (mainly for Ollama)
        if url:
            config["llmProvider"]["url"] = url
            
        # Set parameters if provided
        if parameters:
            config["llmProvider"]["parameters"] = parameters
            
        # Set default model if needed
        if "model" not in config["llmProvider"]:
            config["llmProvider"]["model"] = "llama3"
                
        # Write updated config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            
        return True
    except Exception as e:
        logger.error(f"Failed to update config file: {str(e)}")
        return False

def main():
    """Parse arguments and start the chat session."""
    parser = argparse.ArgumentParser(description="MCP Host for LLM tool interactions")
    parser.add_argument("--config", help="Path to config file (default: config.json in current directory)")
    
    # Model parameters
    parser.add_argument("--model", "-m", help="Override model specified in config")
    parser.add_argument("--message-window", type=int, default=DEFAULT_MESSAGE_WINDOW,
                        help="Number of messages to keep in context")
    
    # Provider selection
    provider_group = parser.add_argument_group("Provider Selection")
    provider_group.add_argument("--provider", choices=["ollama"], 
                              help="Select LLM provider")
    
    # Ollama-specific options
    ollama_group = parser.add_argument_group("Ollama Options")
    ollama_group.add_argument("--ollama-url", help="URL for Ollama API (e.g., http://localhost:11434)")
    ollama_group.add_argument("--ollama-model", help="Ollama model to use (e.g., llama3, mistral, etc.)")
    
    # Other options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--save-config", action="store_true", 
                       help="Save provider options to config file")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Determine provider overrides based on command line arguments
    provider_overrides = {}
    
    # If provider is explicitly set, use it
    if args.provider:
        provider_overrides["type"] = args.provider
    
    # Check for Ollama-specific options
    if args.ollama_url or args.ollama_model:
        # If we have Ollama options but no provider specified, assume Ollama
        if "type" not in provider_overrides:
            provider_overrides["type"] = "ollama"
            
        if args.ollama_url:
            provider_overrides["url"] = args.ollama_url
            
        if args.ollama_model:
            provider_overrides["model"] = args.ollama_model
    
    # Update config file if requested
    if args.save_config and provider_overrides:
        config_path = args.config or os.path.join(os.getcwd(), "config.json")
        provider_type = provider_overrides.get("type")
        model = provider_overrides.get("model") or args.model
        url = provider_overrides.get("url")
        
        if update_config_with_provider_info(config_path, provider_type, model, url):
            print(f"Updated config file with provider settings: {provider_type}")
    
    # Run the chat session
    asyncio.run(run_chat_session(
        config_path=args.config,
        model=args.model,
        message_window=args.message_window,
        provider_overrides=provider_overrides if provider_overrides else None
    ))

if __name__ == "__main__":
    main()