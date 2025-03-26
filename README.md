# MCP Host

A Python implementation of a Model Context Protocol (MCP) host that connects to Ollama LLM backends and MCP servers.

## Features

- **Multiple Server Support**: Connect to any number of MCP-compatible servers
- **Multiple Transport Types**: Supports both stdio and SSE transports
- **Ollama Integration**: Seamless connection to local Ollama models
- **Tool Execution**: Enable LLMs to use tools from connected servers
- **Simple CLI**: Easy-to-use command-line interface
- **JSON Configuration**: Simple config file for server and LLM setup

## Requirements

- Ollama running locally or on a remote server

## Run the MCP Host

   ```bash
   python mcp_host.py
   ```

## Run the weather server(SSE)

   ```bash
   python weather.py
   ```

## Configuration

The MCP Host uses a JSON configuration file to define:

1. **MCP Servers**: The servers that provide tools and resources
2. **LLM Provider**: Configuration for the Ollama backend

### Server Configuration

Each server needs:

- **type**: The transport mechanism (`stdio` or `sse`)
- For stdio servers:
  - **command**: The command to run
  - **args**: Command-line arguments (optional)
  - **env**: Environment variables (optional)
- For SSE servers:
  - **url**: The SSE endpoint URL

### LLM Provider Configuration

- **type**: The provider type (currently only `ollama` is supported)
- **model**: The model name to use (e.g., `llama3`, `mistral`, etc.)
- **url**: The Ollama API URL (default: `http://localhost:11434`)
- **parameters**: Additional parameters for Ollama (temperature, top_p, etc.)

## Command-Line Options

```bash
usage: mcp_host.py [-h] [--config CONFIG] [--model MODEL]
                  [--message-window MESSAGE_WINDOW]
                  [--provider {ollama}] [--ollama-url OLLAMA_URL]
                  [--ollama-model OLLAMA_MODEL] [--debug] [--save-config]

MCP Host for LLM tool interactions

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to config file (default: config.json in current directory)
  --model MODEL, -m MODEL
                        Override model specified in config
  --message-window MESSAGE_WINDOW
                        Number of messages to keep in context

Provider Selection:
  --provider {ollama}   Select LLM provider

Ollama Options:
  --ollama-url OLLAMA_URL
                        URL for Ollama API (e.g., http://localhost:11434)
  --ollama-model OLLAMA_MODEL
                        Ollama model to use (e.g., llama3, mistral, etc.)

Other options:
  --debug               Enable debug logging
  --save-config         Save provider options to config file
```

## Usage Examples

### Basic Usage

```bash
python mcp_host.py
```

### Debug Mode

```bash
python mcp_host.py --debug
```

## Special Commands

During a chat session, you can use the following special commands:

- `tools`: List all available tools from connected servers
- `servers`: List all connected MCP servers
- `exit` or `quit`: End the session

## How It Works

1. When you start MCP Host, it connects to all configured MCP servers
2. Each server provides a list of available tools
3. When you enter a query, it's sent to the Ollama LLM
4. If the LLM decides to use tools, MCP Host executes those tool calls
5. The results are sent back to the LLM
6. The LLM provides a final response incorporating the tool results
