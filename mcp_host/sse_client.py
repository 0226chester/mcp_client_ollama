import os
import json
import re
import asyncio
import time
import uuid
import logging
from typing import Dict, Any, Optional
import aiohttp
from aiohttp.client_exceptions import ClientError

logger = logging.getLogger("mcp-host")

class SSEClient:
    """Client for communicating with an MCP server via Server-Sent Events (SSE)."""
    
    def __init__(self, name: str, url: str):
        self.name = name
        self.sse_url = url
        self.base_url = url.rstrip('/').rsplit('/sse', 1)[0]
        self.session = None
        self.response = None
        self.event_queue = asyncio.Queue()
        self.endpoint_ready = asyncio.Event()
        self.initialized = False
        self._sse_task = None
        self.session_id = f"mcp-host-{uuid.uuid4()}"
        self.server_endpoint = None
        logger.debug(f"Created SSEClient {name} with initial session_id: {self.session_id}")
    
    async def connect(self):
        """Establish SSE connection to the server."""
        try:
            # Create a new session
            self.session = aiohttp.ClientSession()
            
            # Reset the endpoint event flag
            self.endpoint_ready.clear()
            
            # Connect to the SSE endpoint with our initial session ID
            sse_url_with_session = f"{self.sse_url}?session_id={self.session_id}"
            logger.debug(f"Connecting to SSE endpoint: {sse_url_with_session}")
            
            self.response = await self.session.get(sse_url_with_session)
            
            if self.response.status != 200:
                error_text = await self.response.text()
                raise RuntimeError(f"Server returned error: {self.response.status} - {error_text}")
            
            logger.info(f"Connected to SSE endpoint for server: {self.name}")
            
            # Start the event processing task
            self._sse_task = asyncio.create_task(self._process_sse_events())
            
            # Wait for the server to send us the endpoint event
            logger.debug(f"Waiting for endpoint event from server...")
            try:
                await asyncio.wait_for(self.endpoint_ready.wait(), timeout=10.0)
                logger.info(f"Received server endpoint: {self.server_endpoint}")
                return True
            except asyncio.TimeoutError:
                logger.error(f"Timed out waiting for endpoint event from {self.name}")
                return False
        except Exception as e:
            logger.error(f"Failed to connect to SSE endpoint for {self.name}: {str(e)}")
            await self.disconnect()
            return False
    
    async def _process_sse_events(self):
        """Process SSE events from the event stream."""
        event_type = None
        event_data = None
        
        async for line_bytes in self.response.content:
            try:
                line = line_bytes.decode('utf-8').rstrip()
                logger.debug(f"SSE line: {line}")
                
                # Empty line marks end of event
                if not line:
                    if event_type and event_data:
                        await self._handle_event(event_type, event_data)
                    
                    # Reset for next event
                    event_type = None
                    event_data = None
                    continue
                
                # Parse event type
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    continue
                
                # Parse event data
                if line.startswith("data:"):
                    event_data = line[5:].strip()
                    continue
                
                # Handle ping/comment lines that start with :
                if line.startswith(":"):
                    # Just a keep-alive or comment, ignore
                    continue
                
            except asyncio.CancelledError:
                logger.debug(f"SSE event processing task cancelled for {self.name}")
                raise
            except Exception as e:
                logger.error(f"Error processing SSE line: {str(e)}")
    
    async def _handle_event(self, event_type: str, event_data: str):
        """Handle a complete SSE event."""
        logger.debug(f"Handling event: type={event_type}, data={event_data}")
        
        if event_type == "endpoint":
            # This is the critical endpoint event that gives us the URL to use
            self.server_endpoint = event_data
            
            # Extract session_id from the endpoint URL if present
            match = re.search(r'session_id=([^&\s]+)', event_data)
            if match:
                self.session_id = match.group(1)
                logger.debug(f"Extracted session_id from endpoint: {self.session_id}")
            
            logger.debug(f"Setting endpoint ready flag for {self.name}")
            self.endpoint_ready.set()
            
        elif event_type == "message":
            # Regular message, try to parse as JSON
            try:
                message_obj = json.loads(event_data)
                await self.event_queue.put(message_obj)
            except json.JSONDecodeError:
                logger.error(f"Received invalid JSON in message event: {event_data}")
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to the server using the endpoint URL."""
        if not self.session:
            raise RuntimeError(f"No session for {self.name}")
        
        if not self.server_endpoint:
            raise RuntimeError(f"No endpoint URL received from server {self.name}")
        
        try:
            # Construct the full URL for the message endpoint
            if self.server_endpoint.startswith('/'):
                # It's a relative URL, so prepend the base URL
                message_url = f"{self.base_url}{self.server_endpoint}"
            else:
                # It's an absolute URL, use as is
                message_url = self.server_endpoint
            
            logger.debug(f"Sending message to {message_url}: {json.dumps(message)}")
            
            async with self.session.post(message_url, json=message) as response:
                if response.status not in [200, 202]:
                    error_text = await response.text()
                    raise RuntimeError(f"Server returned error: {response.status} - {error_text}")
                
                # For JSON-RPC requests, we need to wait for the response in the SSE stream
                if "id" in message:
                    return await self._wait_for_response(message["id"])
                
                return {}
        except Exception as e:
            logger.error(f"Error sending message to {self.name}: {str(e)}")
            raise
    
    async def _wait_for_response(self, request_id: Any, timeout: float = 30.0) -> Dict[str, Any]:
        """Wait for a response with a matching ID from the SSE event stream."""
        try:
            start_time = time.time()
            logger.debug(f"Waiting for response to request {request_id}")
            
            while time.time() - start_time < timeout:
                try:
                    # Wait for a message with a short timeout
                    try:
                        event = await asyncio.wait_for(self.event_queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        # No message yet, keep waiting
                        continue
                    
                    # Check if this is a notification
                    if "id" not in event and "method" in event:
                        logger.debug(f"Received notification: {event['method']}")
                        # Process notification if needed
                        continue
                    
                    # Check if this matches our request
                    if "id" in event and event["id"] == request_id:
                        logger.debug(f"Found matching response for request {request_id}")
                        return event
                    
                    # If we get here, we received a response for a different request
                    # Put it back in the queue for other waiters
                    await self.event_queue.put(event)
                    
                    # Wait a bit before checking again
                    await asyncio.sleep(0.1)
                except asyncio.QueueEmpty:
                    # Wait a bit before checking again
                    await asyncio.sleep(0.1)
            
            raise TimeoutError(f"Timeout waiting for response to request {request_id}")
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.error(f"Timeout waiting for response from {self.name}")
            else:
                logger.error(f"Error waiting for response from {self.name}: {str(e)}")
            raise
            
    async def disconnect(self):
        """Close the SSE connection according to the MCP HTTP shutdown protocol."""
        logger.debug(f"Disconnecting SSE client for {self.name}")
        
        if self._sse_task and not self._sse_task.done():
            logger.debug(f"Cancelling SSE event processing task for {self.name}")
            self._sse_task.cancel()
            try:
                # Add a timeout to prevent hanging
                await asyncio.wait_for(asyncio.shield(self._sse_task), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.debug(f"SSE task cancel timeout or cancelled for {self.name}")
            except Exception as e:
                logger.error(f"Error cancelling SSE task for {self.name}: {str(e)}")
            
            self._sse_task = None
        
        if self.response:
            logger.debug(f"Closing SSE response for {self.name}")
            try:
                # Use a safer response close method
                self.response.close()
            except Exception as e:
                logger.debug(f"Error closing SSE response (non-critical): {str(e)}")
            self.response = None
        
        if self.session:
            logger.debug(f"Closing HTTP session for {self.name}")
            try:
                # Use timeout with session close
                await asyncio.wait_for(self.session.close(), timeout=2.0)
            except asyncio.TimeoutError:
                logger.warning(f"Timeout closing HTTP session for {self.name}")
            except Exception as e:
                logger.debug(f"Error closing HTTP session (non-critical): {str(e)}")
                
            self.session = None
        
        logger.info(f"Disconnected from SSE server: {self.name}")