import logging
from typing import Dict, List, Any

from .config import ServerConfig
from .sse_client import SSEClient
from .stdio_client import StdioClient

logger = logging.getLogger("mcp-host")

class MCPClient:
    """Client for interacting with an MCP server."""
    
    def __init__(self, name: str, config: ServerConfig):
        self.name = name
        self.config = config
        self.transport = None
        self.initialized = False
        self.next_id = 1
    
    async def initialize(self):
        """Initialize the MCP connection."""
        # Initialize the appropriate transport
        if self.config.type == "stdio":
            if not self.config.command:
                logger.error(f"Server {self.name} is missing required 'command' for stdio transport")
                return False
            
            self.transport = StdioClient(
                name=self.name,
                command=self.config.command,
                args=self.config.args or [],
                env=self.config.env or {}
            )
        elif self.config.type == "sse":
            if not self.config.url:
                logger.error(f"Server {self.name} is missing required 'url' for SSE transport")
                return False
            
            self.transport = SSEClient(
                name=self.name,
                url=self.config.url
            )
        else:
            logger.error(f"Unsupported transport type for server {self.name}: {self.config.type}")
            return False
        
        # Connect using the selected transport
        if not await self.transport.connect():
            return False
        
        # Initialize the MCP connection
        init_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "mcphost-python",
                    "version": "0.1.0"
                },
                "capabilities": {}
            }
        }
        
        try:
            response = await self.transport.send_message(init_request)
            
            if response and "result" in response:
                # Send 'initialized' notification
                initialized_notification = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                
                try:
                    await self.transport.send_message(initialized_notification)
                    logger.debug(f"Sent initialized notification to {self.name}")
                except Exception as e:
                    logger.error(f"Failed to send initialized notification to {self.name}: {str(e)}")
                    
                self.initialized = True
                logger.info(f"Initialized MCP server: {self.name}")
                return True
            else:
                logger.error(f"Failed to initialize MCP server {self.name}: Invalid response")
                await self.shutdown()
                return False
        except Exception as e:
            logger.error(f"Failed to initialize MCP server {self.name}: {str(e)}")
            await self.shutdown()
            return False
    
    def _get_next_id(self) -> int:
        """Get the next request ID."""
        request_id = self.next_id
        self.next_id += 1
        return request_id
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server."""
        if not self.initialized:
            await self.initialize()
            
        list_tools_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/list",
            "params": {}
        }
        
        try:
            response = await self.transport.send_message(list_tools_request)
            
            if response and "result" in response and "tools" in response["result"]:
                tools = response["result"]["tools"]
                logger.info(f"Retrieved {len(tools)} tools from {self.name}")
                return tools
            else:
                logger.error(f"Failed to list tools from {self.name}: Invalid response")
                return []
        except Exception as e:
            logger.error(f"Error listing tools from {self.name}: {str(e)}")
            return []
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool on the server."""
        if not self.initialized:
            await self.initialize()
            
        call_tool_request = {
            "jsonrpc": "2.0",
            "id": self._get_next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        
        try:
            response = await self.transport.send_message(call_tool_request)
            
            if response and "result" in response:
                return response["result"]
            elif response and "error" in response:
                error_msg = response["error"].get("message", "Unknown error")
                logger.error(f"Tool execution error: {error_msg}")
                return {"error": error_msg}
            else:
                logger.error(f"Failed to call tool {tool_name}: Invalid response")
                return {"error": "Invalid response from server"}
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Close the connection and shut down the server.
        
        According to the MCP specification, shutdown should use the underlying
        transport mechanism rather than a specific shutdown message.
        """
        # No need to send a shutdown message as per MCP specification
        # Just disconnect the transport
        if self.transport:
            await self.transport.disconnect()
            
        self.initialized = False
        logger.info(f"Shut down MCP server: {self.name}")