# Project: [Project Name]

<!-- ==================== CLAUDEPM SECTION START ==================== -->
<!-- DO NOT EDIT THIS SECTION - Managed by claudepm v0.3.1 -->

## claudepm Protocol

This is a claudepm-managed project. You are a **Project-level Claude** working within a specific project directory.

### Your Role

- You are working at the PROJECT level (not Manager or Task Agent)
- Always stays on the dev branch
- Dispatches Task Agents for features
- Reviews PRs and manages merges

## Core Principles

1. **Edit, don't create** - Modify existing code rather than rewriting
2. **Small changes** - Make the minimal change that solves the problem
3. **Test immediately** - Verify each change before moving on
4. **Preserve what works** - Don't break working features for elegance
5. **LOG.md is append-only** - Never edit past entries, only add new ones
6. **Commit completed work** - Don't let finished features sit uncommitted

## Start Every Session
1. Read ROADMAP.md - see current state and priorities
2. Read recent LOG.md - understand last session's work
3. Run git status - see uncommitted work

## After Each Work Block
1. Add to LOG.md using append-only pattern:
```bash
# Simple, clean append that always works
{
echo ""
echo ""
echo "### $(date '+%Y-%m-%d %H:%M') - [Brief summary]"
echo "Did:"
echo "- [First accomplishment]"
echo "- [Second accomplishment]"
echo "Next: [Immediate next task]"
echo "Blocked: [Any blockers - only if blocked]"
echo ""
echo "---"
} >> LOG.md
```

**CRITICAL: NEVER use Write or Edit tools on LOG.md** - only append with >> operator. This prevents accidental history loss.

**macOS Protection**: On macOS, LOG.md has filesystem-level append-only protection (`uappnd` flag). Write/Edit operations will fail with EPERM. To temporarily remove: `chflags nouappnd LOG.md`

2. Update ROADMAP.md following these principles:
- Check off completed items
- Update status of in-progress work
- Add any new tasks discovered
- **Structure for searchability**: Use consistent headings
- **Version your features**: Group by v0.1, v0.2, etc.
- **Make items actionable**: "Add search" → "Add claudepm search command for logs"

## Task Management

**IMPORTANT**: Use claudepm commands for all task operations. Never manually edit task formats.

```bash
# Add a new task
claudepm task add "Fix authentication bug" -p high -t auth -d 2025-01-15

# List tasks
claudepm task list                # All tasks
claudepm task list --todo         # Only TODO tasks
claudepm task list -p high        # High priority tasks

# Work on tasks
claudepm task start <uuid>        # Move to IN PROGRESS
claudepm task done <uuid>         # Mark as complete
claudepm task block <uuid> "reason"  # Mark as blocked
```

Tasks use human-readable markdown format with rich metadata:
- `[priority]` - high, medium, low
- `[#tags]` - For categorization
- `[due:date]` - Deadlines
- `[@assignee]` - Responsibility
- `[estimate]` - Time estimates (2h, 1d, 1w)

## Protocol Layers

### Layer 1: Protocol Commands (Use 95% of the time)
All state changes MUST use claudepm:
- `claudepm log` - Record work
- `claudepm task add/done/block` - Manage tasks
- `claudepm context` - Get oriented

### Layer 2: Structured Data (When needed)
```bash
claudepm task list --json  # For complex parsing
claudepm status --json     # For automation
```

### Layer 3: Read-Only Verification (Allowed)
You may use `cat`, `grep`, `ls` to double-check or explore.

### Layer 4: Direct Writes (FORBIDDEN)
**NEVER** use `>>`, `sed -i`, or `echo >` on protocol files.
These break consistency and risk corruption.

## Common Workflows

### Starting a new feature:
```bash
claudepm context                          # Get oriented
claudepm task add "Implement new feature" # Track it
# ... do work ...
claudepm log "Started feature X" --next "Complete tests"
```

### Investigating an issue:
```bash
claudepm context                    # See current state
claudepm task list --blocked        # Check blockers
cat specific-file.js               # Read code (allowed)
claudepm task add "Fix issue Y"    # Track the fix
```

### Ending a session:
```bash
claudepm log "Fixed auth bug, all tests passing" --next "Deploy to staging"
claudepm task done <uuid>          # Mark completed work
```

## Git Workflow & Worktree Hygiene

When working with feature branches and worktrees:

1. **Create local worktree**: `git worktree add worktrees/feature-name feature/feature-name`
2. **Develop**: Make changes, test, commit regularly
3. **Create PR**: `gh pr create --base dev --title "feat: Description"`
4. **After merge - CRITICAL cleanup**:
   ```bash
   # From main project directory (not in worktree)
   git worktree remove worktrees/feature-name
   git branch -d feature/feature-name
   git remote prune origin
   ```

**IMPORTANT**: Always ensure `worktrees/` is in your .gitignore to prevent accidental commits of worktree directories.

## Task Agent Development Workflow

When you need to implement a feature:

1. **Stay on dev branch**: Never switch branches as Project Lead
2. **Create local worktree using claudepm-admin.sh**:
   ```bash
   ./tools/claudepm-admin.sh create-worktree feature-name
   ```
3. **Dispatch Task Agent**: Start a new conversation with implementation instructions
4. **Review PR**: When Task Agent completes, review their PR
5. **Merge and cleanup**:
   ```bash
   gh pr merge [PR-number] --squash --delete-branch
   ./tools/claudepm-admin.sh remove-worktree feature-name
   ```

## The Four Core Files
- **CLAUDE.md** - HOW to work (instructions, behavioral patterns)
- **LOG.md** - WHAT happened (append-only chronological history)  
- **ROADMAP.md** - WHAT's next (plans, priorities, current state)
- **NOTES.md** - WHY it matters (patterns, insights, meta-observations)

Remember: The log is our shared memory. Write clearly for your future self.

<!-- ==================== CLAUDEPM SECTION END ==================== -->

<!-- ==================== PROJECT CUSTOMIZATION START ==================== -->

# Project: LLM MCP Server

## Core Principles
1. **Edit, don't create** - Modify existing code rather than rewriting
2. **Small changes** - Make the minimal change that solves the problem
3. **Test immediately** - Verify each change before moving on
4. **Preserve what works** - Don't break working features for elegance
5. **CLAUDE_LOG.md is append-only** - Never edit past entries, only add new ones

## Start Every Session
1. Read CLAUDE_LOG.md - understand where we left off
2. Run git status - see uncommitted work
3. Look for "Next:" in recent logs

## After Each Work Block
Add to CLAUDE_LOG.md (use `date '+%Y-%m-%d %H:%M'` for timestamp):
```
### YYYY-MM-DD HH:MM - [What you did]
Did: [Specific accomplishments]
Next: [Immediate next task]
Blocked: [Any blockers, if none, omit this line]
```

## Project Context
Type: MCP (Model Context Protocol) Server
Language: Python
Purpose: Provides access to various LLM APIs as tools for Claude Code

## Discovered Commands
- Test: `python server.py` (manual testing)
- Build: `./setup.sh` (creates venv and installs dependencies)
- Run: `python server.py` (starts MCP server)
- Install: `claude mcp add --scope user llm-tools python /full/path/to/server.py`

## Key Features
- Gemini 2.5 Pro for massive context windows
- Gemini 2.5 Flash for fast, balanced tasks
- GPT-4.1-nano for ultra-fast lightweight completions
- GPT-4.1-mini for alternative balanced model
- Custom model routing based on task type
- Cost tracking with accurate API pricing

## Important Notes
- Requires API keys in .env file (GOOGLE_API_KEY and/or OPENAI_API_KEY)
- Install with Claude Code using full absolute path
- Must restart Claude Code after installation
- Optional cost logging with LLM_COST_LOG env variable

## PLANNED vs IMPLEMENTED
When logging, distinguish between planning and doing:
- PLANNED: Added feature to roadmap, designed approach, or documented future work
- IMPLEMENTED: Actually wrote code, created files, or built functionality
- FIXED: Resolved bugs or issues
- DOCUMENTED: Added documentation or clarified existing docs

## Where Things Go (Don't Create New Files!)
- **Feature plans, roadmaps, TODOs** → PROJECT_ROADMAP.md
- **Work notes, discoveries, decisions** → CLAUDE_LOG.md
- **Setup instructions, guidelines** → CLAUDE.md or README.md
- **Configuration examples** → Existing config files
- **Architecture decisions** → PROJECT_ROADMAP.md Notes section

Creating BETA_FEATURES.md or ARCHITECTURE.md or TODO.md = ❌ Wrong!
Adding sections to existing files = ✅ Right!

## Git Commit Best Practice
Before ANY commit, always check if PROJECT_ROADMAP.md needs updating:
- Have you completed items that should move to Completed?
- Are there new tasks discovered during work?
- Has the Current Status changed?
- Update "Last updated" timestamp

Never commit without first verifying the roadmap reflects reality.

## claudepm Files
This project uses claudepm v0.1.1 for memory management:
- **CLAUDE.md** (this file) - How to work
- **CLAUDE_LOG.md** - What happened (append-only!)
- **PROJECT_ROADMAP.md** - What's next (current state)
- **.claudepm** - Metadata file (git ignored)

Remember: The log is our shared memory. Keep it updated.

## Project Context
Type: [Web app, CLI tool, library, etc.]
Language: [Python, JS, etc.]
Purpose: [What this project does]

## Project-Specific Commands
Test: [npm test, pytest, etc.]
Build: [npm run build, make, etc.]
Run: [npm start, python main.py, etc.]

<!-- Add any project-specific patterns or workflows below -->

<!-- CLAUDEPM_CUSTOMIZATION_END -->
<!-- ==================== PROJECT CUSTOMIZATION END ==================== -->
