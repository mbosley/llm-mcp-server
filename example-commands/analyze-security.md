Analyze codebase for security vulnerabilities using Gemini

I'll use Gemini's 1M token context to analyze your entire codebase for security issues.

```python
# Use the analyze_with_gemini tool to scan all code
result = analyze_with_gemini(
    prompt="""Analyze this codebase for security vulnerabilities:
    1. SQL injection risks
    2. XSS vulnerabilities  
    3. Authentication/authorization flaws
    4. Hardcoded secrets or API keys
    5. Dependency vulnerabilities
    6. Input validation issues
    
    For each issue found, provide:
    - File and line number
    - Severity (Critical/High/Medium/Low)
    - Recommended fix
    """,
    files=["**/*.py", "**/*.js", "**/*.ts", "**/*.java"]
)
```

The analysis will cover all source files and provide a comprehensive security report.