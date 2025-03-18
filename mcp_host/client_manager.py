import asyncio
import logging
from typing import Dict, List, Any

from .config import ServerConfig
from .sse_client import SSEClient
from .stdio_client import StdioClient

logger = logging.getLogger("mcp-host")

class MCPClientManager:
    """Manage multiple MCP clients."""
    
    def __init__(self, server_configs: Dict[str, ServerConfig]):
        self.server_configs = server_configs
        self.clients = {}
        self.cached_tools = None
        
    async def initialize_clients(self):
        """Initialize all configured MCP clients."""
        for name, config in self.server_configs.items():
            from .mcpclient import MCPClient
            client = MCPClient(name, config)
            success = await client.initialize()
            if success:
                self.clients[name] = client
        
        logger.info(f"Initialized {len(self.clients)} MCP clients")
        return len(self.clients) > 0
        
    async def get_all_tools(self, skip_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get all tools from all servers with namespaced names.
        
        Args:
            skip_refresh: If True, return cached tools instead of refreshing from servers
        """
        # Return cached tools if requested and available
        if skip_refresh and self.cached_tools is not None:
            logger.debug(f"Using cached tools ({len(self.cached_tools)})")
            return self.cached_tools
            
        all_tools = []
        
        for name, client in self.clients.items():
            try:
                tools = await client.list_tools()
                for tool in tools:
                    # Namespace the tool name with the server name
                    tool_with_namespace = tool.copy()
                    tool_with_namespace["name"] = f"{name}__{tool['name']}"
                    all_tools.append(tool_with_namespace)
                logger.info(f"Got {len(tools)} tools from {name}")
            except Exception as e:
                logger.error(f"Failed to get tools from {name}: {str(e)}")
                # Continue with other clients instead of failing entirely
        
        logger.info(f"Collected {len(all_tools)} tools from all servers")
        
        # Cache the tools for future use
        self.cached_tools = all_tools
        
        return all_tools
        
    async def call_tool(self, namespaced_tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool by its namespaced name."""
        try:
            server_name, tool_name = namespaced_tool_name.split("__", 1)
            
            client = self.clients.get(server_name)
            if not client:
                return {"error": f"Server {server_name} not found"}
            
            return await client.call_tool(tool_name, arguments)
        except ValueError:
            return {"error": f"Invalid tool name format: {namespaced_tool_name}"}
        except Exception as e:
            logger.error(f"Error calling tool {namespaced_tool_name}: {str(e)}")
            return {"error": str(e)}
            
    async def shutdown_all(self):
        """Shut down all MCP clients."""
        logger.info("Shutting down all MCP clients...")
        
        # Make a copy of clients to avoid modification during iteration
        clients_to_shutdown = list(self.clients.items())
        
        shutdown_tasks = []
        for name, client in clients_to_shutdown:
            try:
                task = asyncio.create_task(client.shutdown())
                shutdown_tasks.append(task)
            except Exception as e:
                logger.error(f"Error creating shutdown task for {name}: {str(e)}")
        
        if shutdown_tasks:
            done, pending = await asyncio.wait(
                shutdown_tasks, 
                timeout=10.0,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Handle any pending tasks
            for task in pending:
                logger.warning(f"Shutdown task didn't complete in time, cancelling")
                task.cancel()
        
        # Clear clients dictionary
        self.clients.clear()
        
        logger.info("All MCP clients have been shut down")