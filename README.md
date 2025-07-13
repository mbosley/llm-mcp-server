# LLM MCP Server

An MCP (Model Context Protocol) server that provides access to various LLM APIs as tools for Claude Code.

## Features

- **Kimi K2** - State-of-the-art 1T parameter MoE model with 128k context window
- **Gemini 2.5 Pro** - Handle massive context windows for comprehensive analysis
- **Gemini 2.5 Flash** - Fast, capable model for balanced tasks
- **GPT-4.1-nano** - Ultra-fast, lightweight completions for simple tasks
- **GPT-4.1-mini** - Alternative balanced model with good performance
- **Custom model routing** - Choose the right model for each task

## Complete Installation Guide

### Step 1: Clone or Download
```bash
cd ~/projects  # or your preferred directory
git clone [repository-url] llm-mcp-server
cd llm-mcp-server
```

### Step 2: Run Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Create a `.env` file from `.env.example` template
- Make the server executable

### Step 3: Add Your API Keys
Edit the `.env` file:
```bash
nano .env  # or use your preferred editor
```

Add at least one API key (you don't need all):
```
# Anthropic API for Claude models
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# OpenAI API for GPT models  
OPENAI_API_KEY=your_openai_api_key_here

# Google API for Gemini models
GOOGLE_API_KEY=your_google_api_key_here

# Moonshot API for Kimi K2 models
MOONSHOT_API_KEY=your_moonshot_api_key_here

# Optional: Log costs to file (JSON lines format)
# LLM_COST_LOG=/path/to/llm_costs.jsonl
```

### Step 4: Configure Claude Code
In Claude Code, add the MCP server using this command:
```bash
claude mcp add --scope user llm-tools python /full/path/to/llm-mcp-server/server.py
```

For example:
```bash
claude mcp add --scope user llm-tools python /Users/yourname/projects/llm-mcp-server/server.py
```

### Step 5: Verify Installation
Check that it's configured:
```bash
/mcp
```
You should see "llm-tools" listed.

### Step 6: Restart Claude Code
- Completely quit Claude Code (Cmd+Q on Mac, Alt+F4 on Windows)
- Reopen Claude Code

### Step 7: Test the Tools
In any Claude conversation, the tools are now available. Test by asking:
- "Use analyze_with_gemini to check if the tools are working"
- "Use check_costs to see the cost tracking"

## Available Tools

### analyze_with_gemini
Analyze large codebases or documents with Gemini 2.5 Pro's massive context window.
```
Parameters:
- prompt: The analysis query
- files: List of file paths to include in context (optional)
- context: Direct context string (optional)
```

### quick_gpt
Fast responses using GPT-4.1-nano for simple tasks.
```
Parameters:
- prompt: The task or question
- temperature: Control randomness (0-1, default 0.3)
```

### balanced_llm
Use Gemini 2.5 Flash or GPT-4.1-mini for balanced tasks.
```
Parameters:
- prompt: The task or question
- model: "gemini-flash" or "gpt-mini" (default: "gemini-flash")
- max_tokens: Maximum response length (default 1000)
```

### route_to_best_model
Automatically choose the best model based on the task.
```
Parameters:
- prompt: The task description
- task_type: "analysis", "generation", "simple", or "auto"
```

### kimi_chat
Flexible Kimi/Moonshot chat interface supporting simple prompts, multi-turn conversations, and partial pre-filling.
```
Parameters (all optional):
- prompt: Simple prompt for single-turn, or latest user message
- messages: Full conversation history array (overrides prompt if provided)
  - Each message: {"role": "system/user/assistant", "content": "...", "name": "optional", "partial": true/false}
- system: System message (used only if messages not provided)
- partial_response: Pre-fill the assistant's response to maintain character/format
- model: Model name - Common options:
  - 'kimi-k2-0711-preview' (default) - Kimi K2 instruct model for chat, tool use, structured responses
  - 'kimi-k2-base' - Base model for creative completions (when available via API)
  - 'moonshot-v1-auto' - Automatically selects appropriate context size
  - 'moonshot-v1-128k' - 128k context window
  - Note: Currently only instruct models are available via API
- temperature: Control randomness (0-1, default 0.6)
- max_tokens: Maximum response length (default 4096)
- available_tools: List of built-in tool names to enable (optional)
  - Built-in tools: ["get_current_time", "add_numbers", "list_files"]
- dynamic_tools: Array of custom tool definitions for on-the-fly CLI execution (optional)
  - Each tool needs: name, command, and schema properties
- session_id: Persistent conversation management (optional)
  - New sessions: Use descriptive keywords like "python-async-debugging" (auto-timestamped)
  - Continue existing: Use full ID like "python-async-debugging_20250712_2055"
  - Special commands: "@last" (continue most recent), "@list" (show all sessions), "@clear:id" (delete session)
- return_conversation: Return full conversation history with response (default: false)

Usage modes:
1. Simple: kimi_chat(prompt="Hello")
2. With system: kimi_chat(system="You are a pirate", prompt="Tell me about treasure")
3. Multi-turn: kimi_chat(messages=[...])
4. With pre-filling: kimi_chat(prompt="...", partial_response="*thinks carefully* ")
5. Full conversation + pre-fill: kimi_chat(messages=[...], partial_response="...")
6. With built-in tools: kimi_chat(prompt="What time is it?", available_tools=["get_current_time"])
7. With dynamic tools: kimi_chat(prompt="Check port 8080", dynamic_tools=[{...}])
8. Start new session: kimi_chat(prompt="Let's discuss Python", session_id="python-basics")
9. Continue session: kimi_chat(prompt="Tell me more", session_id="python-basics_20250712_2055")
10. Continue last session: kimi_chat(prompt="What were we discussing?", session_id="@last")
11. List sessions: kimi_chat(session_id="@list")
12. Get conversation state: kimi_chat(prompt="Hello", return_conversation=true)
```

#### Session Management

Kimi conversations can be persisted to disk for long-term continuity:

**Session Storage:**
- Sessions are stored in `.kimi_sessions/` directory (gitignored)
- Each session is a JSON file with messages, metadata, and timestamps
- Session IDs use format: `keywords_YYYYMMDD_HHMM`

**Example Session Workflow:**
```python
# Start a new project discussion
kimi_chat(
    prompt="I need help designing a REST API",
    session_id="api-design"  # Creates: api-design_20250712_2100.json
)

# Continue later (use full ID)
kimi_chat(
    prompt="What about authentication?",
    session_id="api-design_20250712_2100"
)

# Or continue the most recent session
kimi_chat(
    prompt="Should we use JWT or sessions?",
    session_id="@last"
)

# Check all active sessions
kimi_chat(session_id="@list")
```

**Session Features:**
- Automatic message truncation when approaching context limits
- Topic extraction from session IDs for easy identification
- Timestamps for creation and last access
- Session metadata preservation

#### Dynamic Tool Generation

The `kimi_chat` tool supports creating CLI tools on-the-fly using the `dynamic_tools` parameter:

```python
kimi_chat(
    prompt="What's my public IP address?",
    dynamic_tools=[{
        "name": "get_public_ip",
        "command": "curl -s ifconfig.me",
        "schema": {
            "type": "function",
            "function": {
                "name": "get_public_ip",
                "description": "Get public IP address",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    }]
)
```

**Dynamic Tool Examples:**

```python
# Network operations
{
    "name": "check_port",
    "command": "netstat -an | grep :8080",
    "schema": {...}
}

# File operations  
{
    "name": "find_large_files",
    "command": "find . -size +100M -type f -exec ls -lh {} \\;",
    "schema": {...}
}

# System information
{
    "name": "memory_usage", 
    "command": "free -h",
    "schema": {...}
}

# Git operations
{
    "name": "recent_commits",
    "command": "git log --oneline -5", 
    "schema": {...}
}
```

**Tool Execution Features:**
- 30-second timeout for safety
- Stdout/stderr capture  
- Error handling and reporting
- Works with any shell command
- Combines with built-in tools seamlessly

**Complete Dynamic Tool Example:**

```python
# Request: "Check if port 8080 is open and count Python files"
kimi_chat(
    prompt="Check if port 8080 is open and count Python files in this directory",
    dynamic_tools=[
        {
            "name": "check_port_8080",
            "command": "netstat -an | grep :8080",
            "schema": {
                "type": "function",
                "function": {
                    "name": "check_port_8080",
                    "description": "Check if port 8080 is listening",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            }
        },
        {
            "name": "count_python_files", 
            "command": "find . -name '*.py' -type f | wc -l",
            "schema": {
                "type": "function",
                "function": {
                    "name": "count_python_files",
                    "description": "Count Python files in directory",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            }
        }
    ]
)

# Result: 
# "Port 8080 is not currently listening, and there are 3,234 Python files in this directory.
# 
# **Tool Executions:**
# 🔧 check_port_8080({}) → (no output - port not listening)
# 🔧 count_python_files({}) → 3234"
```
```

## Kimi K2 Tool Calling Capabilities

Kimi K2 has native tool-calling capabilities that allow it to autonomously execute functions and orchestrate workflows. The model can:

- **Autonomous Task Decomposition**: Break down complex requests into tool sequences
- **Multiple Tool Calls**: Execute several functions in a single response
- **Parameter Parsing**: Correctly extract and format function arguments
- **Context Preservation**: Maintain conversation state throughout tool interactions

### Tool Calling API Flow

The tool calling process follows OpenAI's tools format:

#### 1. Tool Definition
Tools are defined using JSON schemas:
```json
{
  "type": "function",
  "function": {
    "name": "get_current_time",
    "description": "Get the current date and time",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  }
}
```

#### 2. Tool Call Request
When Kimi wants to use a tool, it responds with:
```json
{
  "role": "assistant",
  "content": "I'll get the current time for you.",
  "tool_calls": [
    {
      "index": 0,
      "id": "get_current_time:0",
      "type": "function",
      "function": {
        "name": "get_current_time",
        "arguments": "{}"
      }
    }
  ]
}
```

#### 3. Tool Execution
Your application executes the function and returns results:
```json
{
  "role": "tool",
  "tool_call_id": "get_current_time:0",
  "content": "2025-07-12 14:30:45"
}
```

#### 4. Final Response
Kimi synthesizes the tool results into natural language:
```json
{
  "role": "assistant",
  "content": "It is currently 2:30:45 PM on July 12, 2025."
}
```

### Example: Multiple Tool Calls

**Request:**
```
"What time is it, and then add 10 plus 5?"
```

**Kimi's Response:**
```json
{
  "tool_calls": [
    {
      "id": "get_current_time:0",
      "function": {"name": "get_current_time", "arguments": "{}"}
    },
    {
      "id": "add_numbers:1", 
      "function": {"name": "add_numbers", "arguments": "{\"a\": 10, \"b\": 5}"}
    }
  ]
}
```

**Tool Results:**
```json
[
  {"role": "tool", "tool_call_id": "get_current_time:0", "content": "2025-07-12 14:45:30"},
  {"role": "tool", "tool_call_id": "add_numbers:1", "content": "15"}
]
```

**Final Response:**
```
"It's currently **2:45:30 PM** on July 12, 2025.
And **10 + 5 = 15**."
```

### Key Implementation Notes

- **Tool Call IDs**: Each tool call has a unique ID (e.g., `"function_name:index"`)
- **Argument Format**: Function arguments are JSON strings, even for simple parameters
- **Multiple Tools**: Kimi can call multiple tools in one response for complex requests
- **Error Handling**: Tool execution errors should be returned as tool messages for Kimi to handle
- **Streaming**: Tool calls work with both streaming and non-streaming API calls

### check_costs
Check cumulative costs for all LLM usage in this session.
```
Parameters: none
Returns: Detailed cost breakdown by model
```

## Usage Examples

In Claude Code, these tools become available for use:

```python
# Analyze entire codebase
result = analyze_with_gemini(
    prompt="Find all API endpoints and their authentication methods",
    files=["src/**/*.py"]
)

# Quick formatting
formatted = quick_gpt(
    prompt="Format this JSON: {a:1,b:2}"
)

# Smart routing
response = route_to_best_model(
    prompt="Explain the authentication flow",
    task_type="analysis"
)

# Kimi/Moonshot examples
# Simple query
answer = kimi_chat(prompt="What is quantum computing?")

# Role-playing with system message
story = kimi_chat(
    system="You are a wise wizard who speaks in riddles",
    prompt="Tell me about dragons"
)

# Multi-turn conversation
chat = kimi_chat(
    messages=[
        {"role": "system", "content": "You are a helpful coding assistant"},
        {"role": "user", "content": "How do I sort a list in Python?"},
        {"role": "assistant", "content": "You can use the sorted() function or list.sort() method."},
        {"role": "user", "content": "What's the difference?"}
    ]
)

# With partial pre-filling for character consistency
response = kimi_chat(
    system="You are Dr. Kelsier from Arknights",
    prompt="What do you think about the current situation?",
    partial_response="*adjusts her monocle with a slight frown* The situation, you ask? "
)

# Using different Moonshot models
auto_response = kimi_chat(
    prompt="Explain quantum computing",
    model="moonshot-v1-auto"  # Automatically selects context size
)
```

## Cost Tracking

Every request automatically tracks:
- Token usage (estimated)
- Cost per request
- Cumulative costs per model
- Total costs across all models

Each response includes cost info:
```
[Response content]

---
💰 Cost: $0.000125 | Total: $0.0045
```

Use `check_costs()` tool anytime to see detailed breakdown.

## Troubleshooting

### Tools not appearing?
1. Make sure you used the full absolute path in the `claude mcp add` command
2. Check `/mcp` shows the server as "Running"
3. Ensure you restarted Claude Code completely
4. Verify at least one API key is set in `.env`

### Getting API errors?
- Check your API keys are valid
- Ensure you have credits/quota on your accounts
- Google API key needs Gemini API enabled in Google Cloud Console
- OpenAI key needs appropriate model access
- Anthropic key needs to be from the Anthropic Console

### Python errors?
- Make sure Python 3.8+ is installed
- Try using the virtual environment: `source venv/bin/activate`
- Check all dependencies installed: `pip list`

### Server not starting?
- Check that server.py has execute permissions: `chmod +x server.py`
- Verify the path in your MCP configuration is correct
- Check Claude Code logs for error messages

## Slash Commands

You can build slash commands that use these tools:

```markdown
# .claude/commands/analyze-all.md
Analyze entire codebase with Gemini

Use Gemini's 1M context to analyze all files:
- Security vulnerabilities
- Architecture patterns
- Code quality issues
```