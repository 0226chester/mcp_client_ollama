import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("mcp-host")

@dataclass
class ContentBlock:
    """Content block for messages."""
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    tool_use_id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Any] = None
    content: Optional[Any] = None

@dataclass
class Message:
    """A message in the conversation."""
    role: str
    content: List[ContentBlock]
    
    def get_text_content(self) -> str:
        """Get the text content from the message."""
        text_blocks = [block.text for block in self.content if block.type == "text" and block.text]
        return " ".join(text_blocks)
    
    def get_tool_calls(self) -> List[Tuple[str, str, Any]]:
        """Get tool calls from the message as (id, name, input) tuples."""
        return [
            (block.id, block.name, block.input) 
            for block in self.content 
            if block.type == "tool_use"
        ]
    
    def is_tool_response(self) -> bool:
        """Check if this message is a tool response."""
        return any(block.type == "tool_result" for block in self.content)
    
    def get_tool_response_id(self) -> Optional[str]:
        """Get the ID of the tool call this message is responding to."""
        for block in self.content:
            if block.type == "tool_result":
                return block.tool_use_id
        return None
    
    def has_content(self) -> bool:
        """Check if message has any meaningful content."""
        return len(self.content) > 0

class ChatSession:
    """Manage a chat session with message history and tool execution."""
    
    def __init__(
        self, 
        llm_provider: Any, 
        mcp_manager: Any,
        message_window: int = 10
    ):
        self.llm_provider = llm_provider
        self.mcp_manager = mcp_manager
        self.message_window = message_window
        self.messages = []
        self.tool_mapping = {}  # Maps non-namespaced tool names to their full namespaced versions
        
    async def refresh_tool_mapping(self):
        """Create a mapping of non-namespaced tool names to their namespaced versions."""
        tools = await self.mcp_manager.get_all_tools()
        self.tool_mapping = {}
        
        for tool in tools:
            # Assuming namespaced tool names are in the format "server__toolname"
            if "__" in tool["name"]:
                server, original_name = tool["name"].split("__", 1)
                # Map both the original name and the namespaced name
                self.tool_mapping[original_name] = tool["name"]
                self.tool_mapping[tool["name"]] = tool["name"]  # Map to itself for completeness
                
        logger.debug(f"Tool mapping refreshed: {self.tool_mapping}")
        return tools
        
    def add_message(self, message: Message):
        """Add a message to the history."""
        if message and message.has_content():
            self.messages.append(message)
            
            # Prune messages if we exceed the window
            if len(self.messages) > self.message_window:
                self.messages = self.messages[-self.message_window:]
    
    async def process_prompt(self, prompt: str) -> str:
        """Process a user prompt and return the response."""
        # Add recursion limit to prevent infinite tool call loops
        return await self._process_prompt_with_limit(prompt, max_iterations=5)
        
    async def _process_prompt_with_limit(self, prompt: str, max_iterations: int = 5, current_iteration: int = 0) -> str:
        """Process a prompt with a limit on tool call iterations."""
        # Check recursion limit to prevent infinite loops
        if current_iteration >= max_iterations:
            logger.warning(f"Reached maximum tool call iterations ({max_iterations}). Forcing final response.")
            # Force a final response without tools to break the loop
            return await self._generate_final_response(f"I've reached the maximum number of tool interactions ({max_iterations}).")
        
        # Step 1: Add user message to history
        if prompt and current_iteration == 0:  # Only add user message on first iteration
            self.add_message(Message(
                role="user",
                content=[ContentBlock(type="text", text=prompt)]
            ))
            
        # Step 2: Get all tools and refresh the mapping, but only on first iteration
        if current_iteration == 0:
            tools = await self.refresh_tool_mapping()
        else:
            # Reuse the same tools to avoid unnecessary API calls
            tools = await self.mcp_manager.get_all_tools(skip_refresh=True)
        
        # Only pass tools when we're not at the last iteration
        tools_to_use = tools if current_iteration < max_iterations - 1 else None
        
        # Step 3: Initial LLM call with the complete message history
        logger.info(f"Making LLM API call (iteration {current_iteration + 1}/{max_iterations})")
        llm_response = await self.llm_provider.create_message(
            messages=self.messages,
            tools=tools_to_use,
            prompt=None  # Already added to messages
        )
        
        # Log the text content for debugging
        text_content = llm_response.get_text_content()
        logger.debug(f"LLM response text: {text_content[:100]}..." if text_content else "No text in response")
        
        # Add the LLM's response to the message history
        self.add_message(llm_response)
        
        # Step 4: Check if LLM wants to use tools
        tool_calls = llm_response.get_tool_calls()
        
        if not tool_calls:
            # No tool calls, just return the LLM's text response
            logger.info("No tool calls in LLM response")
            if not text_content:
                # If there's no text content, generate a fallback response
                return await self._generate_final_response()
            return text_content
            
        logger.info(f"LLM requested {len(tool_calls)} tool call(s)")
        
        # Step 5: Process tool calls and collect results
        for tool_id, tool_name, tool_input in tool_calls:
            logger.info(f"Processing tool call: {tool_name}")
            
            # Parse tool input if it's a string
            if isinstance(tool_input, str):
                try:
                    tool_input = json.loads(tool_input)
                except json.JSONDecodeError:
                    tool_input = {"input": tool_input}
            elif tool_input is None:
                tool_input = {}
            
            # Map the tool name to its namespaced version
            namespaced_tool_name = self._get_namespaced_tool_name(tool_name)
            if not namespaced_tool_name:
                logger.error(f"No matching server found for tool: {tool_name}")
                # Add an error response for this tool
                self.add_message(Message(
                    role="tool", # Use "tool" role instead of "user"
                    content=[ContentBlock(
                        type="tool_result",
                        tool_use_id=tool_id,
                        content=[{
                            "type": "text", 
                            "text": f"Error: Tool '{tool_name}' not found or not available in any connected server."
                        }]
                    )]
                ))
                continue
                
            logger.debug(f"Mapped tool name '{tool_name}' to '{namespaced_tool_name}'")
            
            # Process the tool input to handle null values
            processed_input = self._process_tool_arguments(tool_name, tool_input)
            
            # Call the tool with the proper namespaced name and processed input
            logger.info(f"Calling tool: {namespaced_tool_name}")
            try:
                result = await self.mcp_manager.call_tool(namespaced_tool_name, processed_input)
                
                # Transform the tool result to match expected format
                transformed_content = self._transform_tool_result_content(result)
                
                # Add tool result to message history with "tool" role
                self.add_message(Message(
                    role="tool",
                    content=[ContentBlock(
                        type="tool_result",
                        tool_use_id=tool_id,
                        content=transformed_content
                    )]
                ))
                
                logger.debug(f"Tool result: {transformed_content}")
                
            except Exception as e:
                # Handle tool call errors
                logger.error(f"Error calling tool {namespaced_tool_name}: {str(e)}")
                self.add_message(Message(
                    role="tool",
                    content=[ContentBlock(
                        type="tool_result",
                        tool_use_id=tool_id,
                        content=[{
                            "type": "text", 
                            "text": f"Error executing tool '{tool_name}': {str(e)}"
                        }]
                    )]
                ))
        
        # Step 7: Make a second call to the LLM with the tool results included
        logger.info("Making follow-up LLM call with tool results")
        final_response = await self.llm_provider.create_message(
            messages=self.messages,
            tools=tools_to_use,  # Keep tools available in case more tool calls are needed
            prompt=None
        )
        
        # Add the final response to history
        self.add_message(final_response)
        
        # Check if LLM made more tool calls - if so, we need another round
        if final_response.get_tool_calls() and current_iteration < max_iterations - 1:
            logger.info("LLM made additional tool calls - processing another round")
            # Recursive call, but with no new user prompt and increment iteration
            return await self._process_prompt_with_limit(None, max_iterations, current_iteration + 1)
        
        # Step 8: Return the final text content
        logger.info("Returning final LLM response")
        text_response = final_response.get_text_content()
        
        # If there's no text in the response, generate a simple summary
        if not text_response:
            return await self._generate_final_response()
            
        return text_response
    
    async def _generate_final_response(self, fallback_text=None):
        """Generate a generic final response when the LLM isn't producing text."""
        logger.info("Generating contextual fallback response")
        
        # Case 1: Use explicitly provided fallback text
        if fallback_text:
            return fallback_text
        
        # Case 2: Extract the most recent tool result data
        tool_data = None
        tool_name = None
        
        for msg in reversed(self.messages):
            if msg.role == "tool":
                for block in msg.content:
                    if block.type == "tool_result":
                        # Extract tool name from tool use if possible
                        for i, prev_msg in enumerate(reversed(self.messages[:self.messages.index(msg)])):
                            if prev_msg.role == "assistant":
                                for prev_block in prev_msg.content:
                                    if prev_block.type == "tool_use" and prev_block.id == block.tool_use_id:
                                        tool_name = prev_block.name
                                        break
                                if tool_name:
                                    break
                        
                        # Extract tool result content
                        if isinstance(block.content, list):
                            for content_item in block.content:
                                if isinstance(content_item, dict) and content_item.get("type") == "text":
                                    tool_data = content_item.get("text")
                                    break
                        elif isinstance(block.content, str):
                            tool_data = block.content
                        
                        if tool_data:
                            break
                
                if tool_data:
                    break
        
        # Case 3: Generate a contextual response based on tool data
        if tool_data:
            tool_context = f" using the {tool_name}" if tool_name else ""
            
            # Truncate overly long tool data for the summary
            if len(tool_data) > 500:
                tool_data_summary = tool_data[:500] + "..."
            else:
                tool_data_summary = tool_data
                
            return (f"I retrieved the following information{tool_context}:\n\n"
                    f"{tool_data_summary}")
        
        # Case 4: Completely generic fallback
        prompt_history = ""
        for msg in self.messages[-3:]:  # Look at last 3 messages for context
            if msg.role == "user":
                prompt_text = msg.get_text_content()
                if prompt_text:
                    prompt_history = prompt_text
                    break
        
        if prompt_history:
            return (f"I've processed your request about \"{prompt_history[:50]}{'...' if len(prompt_history) > 50 else ''}\" "
                    f"but I'm having trouble providing a complete response. Could you please clarify or rephrase your question?")
        else:
            return "I'm having trouble generating a response. Could you please provide more details or rephrase your question?"
        
    def _get_namespaced_tool_name(self, tool_name: str) -> Optional[str]:
        """Map a non-namespaced tool name to its namespaced version."""
        # If the tool name is already namespaced, return it
        if "__" in tool_name:
            return tool_name
            
        # Try to find a mapping for this tool name
        return self.tool_mapping.get(tool_name)
        
    def _process_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Process tool arguments to fix common issues before sending to server."""
        if not arguments or not isinstance(arguments, dict):
            return arguments

        # Create a copy to avoid modifying the original
        processed = arguments.copy()
        
        # Handle specific tools that we know about
        if tool_name.endswith("fetch"):
            # Set default values for fetch tool parameters
            if "max_length" in processed and processed["max_length"] is None:
                processed["max_length"] = 5000  # Set a reasonable default
                
            if "start_index" in processed and processed["start_index"] is None:
                processed["start_index"] = 0  # Start from beginning
        
        # General handling for all tools: remove all null values as they often cause validation errors
        processed = {k: v for k, v in processed.items() if v is not None}
            
        logger.debug(f"Processed tool arguments for {tool_name}: {processed}")
        return processed
    
    def _transform_tool_result_content(self, result):
        """Transform tool result content to match expected format."""
        # Case 1: Already in content block list format
        if isinstance(result, dict) and 'content' in result:
            content = result['content']
            if isinstance(content, list):
                # Validate each content block has required fields
                for item in content:
                    if not isinstance(item, dict) or 'type' not in item:
                        # Fix malformed content block
                        return self._create_text_content_blocks(result)
                return content
        
        # Case 2: Result is a string
        if isinstance(result, str):
            return [{"type": "text", "text": result}]
        
        # Case 3: Result is a dictionary (convert to formatted JSON)
        if isinstance(result, dict):
            # Special case: Error response
            if 'error' in result:
                error_msg = result['error']
                return [{"type": "text", "text": f"Error: {error_msg}"}]
            
            # Try to extract text content if available
            if 'text' in result:
                return [{"type": "text", "text": result['text']}]
                
            # Convert entire dict to formatted JSON
            try:
                formatted_json = json.dumps(result, indent=2)
                return [{"type": "text", "text": formatted_json}]
            except Exception as e:
                logger.error(f"Error converting result to JSON: {str(e)}")
        
        # Case 4: Other types (lists, etc.)
        try:
            return [{"type": "text", "text": str(result)}]
        except Exception as e:
            logger.error(f"Error converting result to string: {str(e)}")
            return [{"type": "text", "text": "Error: Unable to process tool result"}]
        
    def _create_text_content_blocks(self, data):
        """Create standard text content blocks from various data formats."""
        if isinstance(data, str):
            return [{"type": "text", "text": data}]
        try:
            return [{"type": "text", "text": json.dumps(data, indent=2)}]
        except:
            return [{"type": "text", "text": str(data)}]