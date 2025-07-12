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

### kimi_k2_base
Use Kimi K2 Base model (1T params) for raw completions and experimentation.
```
Parameters:
- prompt: The input prompt
- temperature: Control randomness (0-1, default 0.6)
- max_tokens: Maximum response length (default 4096)
```

### kimi_k2_instruct
Use Kimi K2 Instruct model for chat, tool use, and agentic tasks with 128k context.
```
Parameters:
- prompt: The user message or task
- system: System message for behavior guidance (optional)
- temperature: Control randomness (0-1, default 0.6)
- max_tokens: Maximum response length (default 4096)
```

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