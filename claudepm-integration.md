# Integrating LLM Tools with claudepm

Once you have the LLM MCP server running, you can create powerful slash commands for claudepm that leverage different models.

## Example claudepm Commands

### 1. Codebase-Wide Analysis
Create `.claude/commands/analyze-all-code.md`:
```markdown
Analyze entire codebase for patterns and issues using Gemini

I'll use Gemini's massive context window to analyze all your code at once.

1. Gather all source files
2. Use analyze_with_gemini to process everything
3. Generate comprehensive report

This can find cross-project patterns, architectural issues, and opportunities for refactoring.
```

### 2. Fast Project Status
Create `.claude/commands/quick-status.md`:
```markdown
Get quick project status using GPT-4o-mini

I'll use GPT-4o-mini to quickly summarize each project's status.

For each project:
- Use quick_gpt to analyze last log entry
- Extract current state and blockers
- Format into concise summary

This is 10x faster than full analysis while still being useful.
```

### 3. Smart Brain Dump
Create `.claude/commands/smart-brain-dump.md`:
```markdown
Process brain dump with AI assistance

I'll use multiple models to intelligently process your brain dump:

1. Use analyze_with_gemini to understand the full context
2. Use claude_haiku to extract action items
3. Use quick_gpt to format updates for each project
4. Route updates using parallel Tasks

This combines the strengths of each model for optimal results.
```

## Architecture Benefits

With the LLM MCP server, claudepm can:

1. **Massive Context Analysis**: Use Gemini to analyze ALL projects at once
2. **Fast Operations**: Use GPT-4o-mini for quick status checks
3. **Intelligent Routing**: Choose the right model for each subtask
4. **Cost Optimization**: Use expensive models only when needed

## Advanced Patterns

### Pattern 1: Hierarchical Analysis
```python
# Manager Claude uses Gemini for overview
overview = analyze_with_gemini(
    prompt="Analyze all projects for common patterns",
    files=["*/PROJECT_ROADMAP.md", "*/CLAUDE_LOG.md"]
)

# Then spawns Tasks with GPT-4o-mini for details
for project in projects:
    Task("Quick check", prompt=f"Use quick_gpt to summarize {project}")
```

### Pattern 2: Smart Routing by Task Size
```python
if len(context) > 50000:
    # Large context -> Gemini
    result = analyze_with_gemini(...)
elif task_is_simple:
    # Simple task -> GPT-4o-mini
    result = quick_gpt(...)
else:
    # Default -> Claude Haiku
    result = claude_haiku(...)
```

### Pattern 3: Model Consensus
```python
# Get opinions from multiple models
gemini_view = analyze_with_gemini(prompt="Should we refactor this?")
claude_view = claude_haiku(prompt="Should we refactor this?") 
gpt_view = quick_gpt(prompt="Should we refactor this?")

# Synthesize consensus
```

## Cost Optimization

- **Gemini 2.5 Pro**: Best for massive context analysis
- **Gemini 2.5 Flash**: Fast and capable for balanced tasks
- **GPT-4.1-nano**: Ultra-fast and cheap for simple tasks
- **GPT-4.1-mini**: Good alternative for balanced tasks

Use the router to automatically optimize costs while maintaining quality!