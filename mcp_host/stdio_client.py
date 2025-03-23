import os
import json
import asyncio
import logging
import time
from typing import Dict, List, Any

logger = logging.getLogger("mcp-host")

class StdioClient:
    """Client for communicating with an MCP server via stdin/stdout."""
    
    def __init__(self, name: str, command: str, args: List[str], env: Dict[str, str] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or {}
        self.process = None
        self.initialized = False
    
    async def connect(self):
        """Start the MCP server process."""
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        
        try:
            # Create a stopping event for the stderr logging task
            self._stderr_stop_event = asyncio.Event()
            
            # Log stderr from the subprocess
            async def log_stderr():
                try:
                    while not self._stderr_stop_event.is_set():
                        # Check if process still exists before reading stderr
                        if not self.process or not self.process.stderr:
                            await asyncio.sleep(0.1)  # Short sleep to avoid busy waiting
                            continue
                            
                        try:
                            stderr_line = await asyncio.wait_for(
                                self.process.stderr.readline(),
                                timeout=0.5  # Short timeout
                            )
                            if not stderr_line:
                                # End of stream
                                break
                            logger.debug(f"[{self.name} stderr] {stderr_line.decode().rstrip()}")
                        except asyncio.TimeoutError:
                            # Timeout is normal, just try again
                            continue
                        except Exception as e:
                            logger.error(f"Error reading stderr from {self.name}: {str(e)}")
                            break
                except asyncio.CancelledError:
                    # Task was cancelled, exit cleanly
                    logger.debug(f"Stderr logging task for {self.name} was cancelled")
                    
            self.process = await asyncio.create_subprocess_exec(
                self.command,
                *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Start stderr logging task
            self._stderr_task = asyncio.create_task(log_stderr())
            
            logger.info(f"Started stdio server: {self.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to start stdio server {self.name}: {str(e)}")
            return False
    
    async def disconnect(self):
        """Terminate the server process according to the MCP stdio shutdown protocol."""
        if not self.process:
            return
            
        try:
            # Signal the stderr logging task to stop
            if hasattr(self, '_stderr_stop_event'):
                self._stderr_stop_event.set()
            
            # Cancel the stderr logging task if it exists
            if hasattr(self, '_stderr_task') and self._stderr_task and not self._stderr_task.done():
                logger.debug(f"Cancelling stderr logging task for {self.name}")
                self._stderr_task.cancel()
                try:
                    await asyncio.wait_for(self._stderr_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling stderr task for {self.name}: {str(e)}")
                    
                self._stderr_task = None
            
            # 1. Close stdin to the child process using safe transport close
            logger.debug(f"Closing stdin for {self.name}")
            if self.process.stdin:
                await self._safe_close_transport(self.process.stdin)
            
            # 2. Wait for the process to exit (with timeout)
            logger.debug(f"Waiting for {self.name} to exit")
            try:
                await asyncio.wait_for(self.process.wait(), timeout=3.0)
                logger.info(f"Server {self.name} exited gracefully")
                self.process = None
                return
            except asyncio.TimeoutError:
                logger.debug(f"Server {self.name} did not exit after stdin close")
            
            # 3. Send SIGTERM if the process doesn't exit
            logger.debug(f"Sending SIGTERM to {self.name}")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
                logger.info(f"Server {self.name} terminated")
                self.process = None
                return
            except asyncio.TimeoutError:
                logger.warning(f"Server {self.name} did not respond to SIGTERM")
            
            # 4. Send SIGKILL if the process still doesn't exit
            logger.warning(f"Sending SIGKILL to {self.name}")
            self.process.kill()
            await self.process.wait()
            logger.info(f"Server {self.name} killed")
        except Exception as e:
            logger.error(f"Error during shutdown of {self.name}: {str(e)}")
        
        self.process = None
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise RuntimeError(f"Server {self.name} is not running")
        
        request_str = json.dumps(message) + "\n"
        logger.debug(f"Sending message to {self.name}: {request_str}")
        self.process.stdin.write(request_str.encode())
        await self.process.stdin.drain()
        
        if "method" in message and "id" not in message:
            logger.debug(f"Sent notification to {self.name}, no response expected")
            return {}
        
        # Store the request ID for matching with responses
        request_id = message.get("id")
        
        # Add timeout to prevent hanging
        start_time = time.time()
        timeout = 10.0  # 10 second timeout
        
        # Keep reading lines until we get a proper response matching our request ID
        while time.time() - start_time < timeout:
            try:
                response_line = await asyncio.wait_for(
                    self.process.stdout.readline(), 
                    timeout=2.0  # Shorter timeout for each read attempt
                )
                
                if not response_line:
                    continue  # Skip empty lines
                    
                try:
                    response = json.loads(response_line.decode())
                    logger.debug(f"Received message from {self.name}: {response}")
                    
                    # Check if this is a notification (no id field)
                    if "id" not in response and "method" in response:
                        # Handle notification separately
                        logger.debug(f"Received notification: {response['method']}")
                        # You might want to store notifications somewhere or process them
                        continue  # Continue waiting for the actual response
                    
                    # Check if this is our response (matching id)
                    if "id" in response and response["id"] == request_id:
                        logger.debug(f"Matched response with request id: {request_id}")
                        return response
                    
                    # If we got here, we received a response for a different request
                    # This should be rare but possible in asynchronous environments
                    logger.warning(f"Received response with non-matching ID. Expected {request_id}, got {response.get('id')}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing message from {self.name}: {str(e)}")
                    logger.debug(f"Raw message content: {response_line}")
                    continue
                    
            except asyncio.TimeoutError:
                # This is just a timeout for a single read attempt, not the overall timeout
                continue
        
        # If we get here, we've timed out waiting for a matching response
        logger.error(f"Timeout waiting for response with ID {request_id} from {self.name}")
        raise TimeoutError(f"Timeout waiting for response from {self.name}")
        
    # Add this in stdio_client.py
    async def _safe_close_transport(self, transport):
        """Safely close transports with Windows compatibility."""
        if transport:
            try:
                transport.close()
            except (ConnectionResetError, ConnectionAbortedError) as e:
                # Windows often has issues with closing connections
                logger.debug(f"Connection reset during close (expected on Windows): {e}")
            except Exception as e:
                logger.error(f"Error closing transport: {e}")