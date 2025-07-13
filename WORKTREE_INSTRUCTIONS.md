# Worktree Instructions for Model-Agnostic Sessions

## Overview
This worktree contains the implementation for model-agnostic session management. Follow these instructions to implement the refactor incrementally while maintaining backward compatibility.

## Working in this Worktree
- **Branch**: `feature/model-agnostic-sessions`
- **Directory**: `worktrees/model-agnostic-sessions`
- **Purpose**: Implement unified session management across all LLM models

## Key Design Decisions (from Kimi conversation)
1. **Unified session format** with model-specific metadata extensions
2. **Flat file storage** in `.llm_sessions/` with symlink views
3. **Incremental refactor** using feature flags (not a v2 branch)
4. **B+ model switching** - allowed but tracked with warnings
5. **Layered API** - unified core with specialized wrappers

## Implementation Phases

### Phase 1: Foundation (Start Here)
1. Create `utils/feature_flags.py` for `LLM_UNIFIED` environment variable
2. Create `session_manager.py` with:
   - Thread-safe file operations
   - Unified session schema
   - Migration utilities
3. Set up `.llm_sessions/` directory structure

### Phase 2: Core Refactoring
1. Add `_chat_unified()` internal function to `server.py`
2. Create model adapters in `adapters/` directory
3. Update existing functions to use unified core when feature flag is enabled

### Phase 3: Advanced Features
1. Implement model switching with metadata tracking
2. Add session search/indexing capabilities
3. Create migration script for existing Kimi sessions

### Phase 4: Testing & Rollout
1. Comprehensive unit and integration tests
2. Performance benchmarking
3. Gradual production rollout with monitoring

## If You Need Help

### Consulting Kimi
If you encounter design decisions or need clarification, continue the existing Kimi conversation:
- **Session ID**: `refactor-model-agnostic-sessions_20250712_2126`
- **Session File**: `.kimi_sessions/refactor-model-agnostic-sessions_20250712_2126.json`

Example questions for Kimi:
- "How should we handle session compression for long conversations?"
- "What's the best approach for the SQLite index schema?"
- "How should we handle tool state across model switches?"

### Using the Session
```python
# From the main project directory
mcp__llm-mcp-server__kimi_chat(
    prompt="Your implementation question here",
    session_id="refactor-model-agnostic-sessions_20250712_2126"
)
```

## Development Workflow
1. Make changes in this worktree
2. Test with `LLM_UNIFIED=1 python server.py`
3. Ensure backward compatibility (test without flag)
4. Commit changes to feature branch
5. Create PR when ready for review

## Important Files
- `IMPLEMENTATION_GUIDE.md` - Detailed step-by-step implementation
- `server.py` - Main server file to modify
- `.gitignore` - Already updated to ignore worktrees/

## Success Criteria
- [ ] All existing functionality works unchanged
- [ ] New unified system works with feature flag
- [ ] Migration preserves all existing sessions
- [ ] Performance overhead < 10ms
- [ ] 95%+ test coverage on new code

## Next Steps
1. Start with creating `utils/feature_flags.py`
2. Test the feature flag system
3. Begin implementing `session_manager.py`
4. Follow the detailed guide in `IMPLEMENTATION_GUIDE.md`

Remember: This is an incremental refactor. The existing system must continue working while you build the new one alongside it.