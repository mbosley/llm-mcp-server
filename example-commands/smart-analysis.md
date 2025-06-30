Smart analysis that routes to the best model

I'll analyze your request and automatically choose the best LLM for the task.

```python
# Let the router decide which model to use
query = $ARGUMENTS

result = route_to_best_model(
    prompt=query,
    task_type="auto"  # Let the router decide based on query complexity
)
```

The router will:
- Use Gemini for large context analysis (>5000 chars or explicit analysis requests)
- Use GPT-4o-mini for simple tasks (<200 chars)
- Use Claude Haiku for balanced tasks requiring intelligence but not massive context

This ensures you get the best model for each specific task.