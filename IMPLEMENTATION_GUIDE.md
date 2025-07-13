# Model-Agnostic Session Implementation Guide

## Overview
This guide provides step-by-step instructions for implementing model-agnostic session management in the LLM MCP Server. If you encounter any design decisions or implementation challenges, consult Kimi using the existing session: `refactor-model-agnostic-sessions_20250712_2126`.

## Prerequisites
- Working in the `worktrees/model-agnostic-sessions` directory
- Feature branch: `feature/model-agnostic-sessions`
- Main branch should remain stable during development

## Phase 1: Foundation Setup

### 1.1 Feature Flag System
Create `utils/feature_flags.py`:

```python
import os
from typing import Optional

class FeatureFlags:
    """Manage feature flags for incremental rollout"""
    
    @staticmethod
    def is_unified_sessions_enabled() -> bool:
        """Check if unified session management is enabled"""
        return os.getenv('LLM_UNIFIED', '').lower() in ('1', 'true', 'yes')
    
    @staticmethod
    def is_legacy_mode() -> bool:
        """Check if forced legacy mode is enabled"""
        return os.getenv('LLM_LEGACY', '').lower() in ('1', 'true', 'yes')
```

### 1.2 Session Manager Module
Create `session_manager.py` with the following structure:

```python
# Key components to implement:
# 1. SessionManager class with thread safety
# 2. Unified session schema
# 3. File locking for concurrent access
# 4. Migration utilities for existing sessions
```

**Implementation Notes:**
- Use the schema discussed with Kimi (see session for details)
- Include fields for model switching metadata
- Implement atomic file operations with temp files
- Add LRU cache for frequently accessed sessions

### 1.3 Directory Structure
```bash
.llm_sessions/
├── sessions/           # Flat storage of session files
├── views/             # Symlink-based views
│   ├── by-model/
│   └── by-date/
└── .index.sqlite      # Optional search index
```

## Phase 2: Core Refactoring

### 2.1 Unified Chat Core
Modify `server.py` to add `_chat_unified()`:

```python
def _chat_unified(
    messages: List[Dict],
    model: str,
    session_id: Optional[str] = None,
    **kwargs
) -> Dict:
    """
    Unified chat implementation that all model-specific functions will use.
    
    Key responsibilities:
    1. Session loading/creation
    2. Model adapter selection
    3. Request/response handling
    4. Session persistence
    """
    # Implementation based on Kimi's recommendations
```

### 2.2 Model Adapters
Create `adapters/` directory with:
- `base_adapter.py` - Abstract base class
- `gemini_adapter.py` - Gemini-specific logic
- `openai_adapter.py` - OpenAI/GPT logic
- `kimi_adapter.py` - Kimi/Moonshot logic
- `anthropic_adapter.py` - Claude logic (if needed)

Each adapter should handle:
- Model-specific request formatting
- Response parsing
- Metadata extraction
- Error handling

### 2.3 Wrapper Functions
Update existing functions to use unified core:

```python
def analyze_with_gemini(text: str, **kwargs) -> str:
    """Optimized for large context analysis"""
    if FeatureFlags.is_unified_sessions_enabled():
        return _chat_unified(
            messages=[{"role": "user", "content": f"Analyze: {text}"}],
            model="gemini-2.5-pro",
            max_tokens=8192,
            **kwargs
        )
    else:
        # Existing implementation
```

## Phase 3: Advanced Features

### 3.1 Model Switching (B+ Approach)
Implement guardrailed model switching:

```python
def switch_model_in_session(
    session_id: str,
    new_model: str,
    reason: Optional[str] = None
) -> Dict:
    """
    Switch models within an existing session with proper tracking.
    
    Guards:
    1. Explicit user consent required
    2. Compatibility warnings
    3. Metadata tracking
    4. System message injection
    """
```

### 3.2 Session Migration
Create migration script for existing Kimi sessions:

```python
def migrate_kimi_sessions():
    """
    Migrate .kimi_sessions/ to unified format.
    
    Steps:
    1. Read each .kimi_sessions/*.json
    2. Transform to unified schema
    3. Save to .llm_sessions/sessions/
    4. Create appropriate views
    5. Verify migration integrity
    """
```

## Testing Strategy

### Unit Tests
Create `tests/test_session_manager.py`:
- Test concurrent access
- Test file locking
- Test session CRUD operations
- Test migration logic

### Integration Tests
Create `tests/test_unified_chat.py`:
- Test each model through unified interface
- Test session persistence
- Test model switching
- Test backward compatibility

## Rollout Plan

### Week 1: Foundation
- [ ] Implement feature flags
- [ ] Create SessionManager
- [ ] Set up directory structure
- [ ] Add migration utilities

### Week 2: Core Integration
- [ ] Implement _chat_unified
- [ ] Create model adapters
- [ ] Update wrapper functions
- [ ] Add dual-write capability

### Week 3: Testing & Polish
- [ ] Comprehensive testing
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] Edge case handling

### Week 4: Production Rollout
- [ ] Enable for internal testing
- [ ] Monitor metrics
- [ ] Gradual user rollout
- [ ] Prepare legacy removal

## Consulting Kimi

When you need guidance or encounter design decisions:

```python
# Use the existing conversation
session_id = "refactor-model-agnostic-sessions_20250712_2126"

# Example questions to ask:
- "How should we handle session compression for long conversations?"
- "What's the best way to implement the SQLite index?"
- "How should we handle tool state across model switches?"
```

## Common Pitfalls to Avoid

1. **Don't break existing functionality** - Always check feature flags
2. **Don't assume file locks work on all systems** - Add fallbacks
3. **Don't forget about Windows compatibility** - Test symlink alternatives
4. **Don't ignore performance** - Benchmark before/after
5. **Don't skip tests** - Aim for >95% coverage on new code

## Success Criteria

- [ ] All existing tests pass
- [ ] New tests cover edge cases
- [ ] Performance within 10ms of original
- [ ] Zero data loss during migration
- [ ] Clean rollback capability
- [ ] Documentation is comprehensive

## Next Steps

1. Start with Phase 1.1 - Create the feature flag system
2. Test feature flags with a simple example
3. Move on to SessionManager implementation
4. Consult Kimi session if you need clarification on any design decisions

Remember: This is an incremental refactor. Keep the existing system working while building the new one alongside it.