# LLM MCP Server

An MCP (Model Context Protocol) server that provides access to various LLM APIs as tools for Claude Code.

## Features

- **Gemini 2.5 Pro** - Handle massive context windows for comprehensive analysis
- **Gemini 2.5 Flash** - Fast, capable model for balanced tasks
- **GPT-4.1-nano** - Ultra-fast, lightweight completions for simple tasks
- **GPT-4.1-mini** - Alternative balanced model with good performance
- **Custom model routing** - Choose the right model for each task

## Installation

1. Install dependencies:
```bash
pip install mcp anthropic google-generativeai openai python-dotenv
```

2. Create `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

3. Configure Claude Code to use this MCP server by adding to your Claude Code settings:
```json
{
  "mcpServers": {
    "llm-tools": {
      "command": "python",
      "args": ["/path/to/llm-mcp-server/server.py"]
    }
  }
}
```

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