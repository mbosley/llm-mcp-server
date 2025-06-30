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