Quick code refactoring suggestions using GPT-4o-mini

I'll use GPT-4o-mini to quickly suggest refactoring for the code you provide.

```python
# Fast refactoring with GPT-4o-mini
code = $ARGUMENTS

suggestion = quick_gpt(
    prompt=f"""Suggest refactoring for this code:

{code}

Focus on:
1. Code clarity
2. Performance improvements
3. Following best practices
4. Reducing complexity

Keep suggestions concise and actionable.""",
    temperature=0.2  # Low temperature for consistent suggestions
)
```

This uses GPT-4o-mini for fast, cheap refactoring suggestions on small code snippets.