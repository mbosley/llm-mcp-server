#!/usr/bin/env python3
"""
LLM MCP Server - Provides access to various LLM APIs as tools
"""

import os
import json
import asyncio
import subprocess
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import glob
from datetime import datetime
from collections import defaultdict
import time

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# LLM Libraries
import anthropic
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# Import unified session components
from utils.feature_flags import FeatureFlags
from utils.prompt_constructor import construct_prompt
from session_manager import SessionManager
from adapters import (
    GeminiAdapter, OpenAIAdapter, KimiAdapter, AnthropicAdapter
)

# Load environment variables
load_dotenv()

# Safety features for CLI tool execution
DANGEROUS_COMMANDS = {
    # File system destruction
    "rm", "rmdir", "del", "format", "fdisk", "dd", "mkfs", "shred",
    # System control
    "shutdown", "reboot", "poweroff", "halt", "init",
    # User/permission management  
    "passwd", "useradd", "userdel", "groupadd", "usermod", "groupmod",
    "chmod", "chown", "chgrp", "umask",
    # Process control
    "kill", "pkill", "killall", "nice", "renice",
    # System configuration
    "systemctl", "service", "update-rc.d", "chkconfig",
    "iptables", "firewall-cmd", "ufw",
    # Package management
    "apt", "apt-get", "yum", "dnf", "pacman", "brew", "npm", "pip",
    # Dangerous utilities
    "eval", "exec", "source", "sudo", "su", "doas",
}

DANGEROUS_PATTERNS = [
    # Shell redirections and pipes that could be destructive
    ">", ">>", ">&", "&>", "2>", "2>>",
    # Command substitution
    "$(", "${", "`",
    # Path traversal and sensitive files
    "..", "~/.ssh", "~/.aws", "/etc/passwd", "/etc/shadow", ".env",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
    # Sudo variations
    "sudo ", "sudo\t", "doas ", "su ",
    # Background execution
    "&", "nohup",
    # Multiple command execution
    "&&", "||", ";", "\n",
]

def is_command_safe(command: str) -> tuple[bool, str]:
    """Check if a command is safe to execute"""
    command_lower = command.lower()
    
    # Check for dangerous base commands
    first_word = command_lower.split()[0] if command_lower.split() else ""
    if first_word in DANGEROUS_COMMANDS:
        return False, f"Command '{first_word}' is not allowed for safety reasons"
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return False, f"Pattern '{pattern}' is not allowed in commands"
    
    # Check for attempts to escape or chain commands
    if any(op in command for op in ["&&", "||", ";", "|", "\n"]):
        return False, "Command chaining is not allowed"
    
    return True, "Command appears safe"

# CLI Tool execution framework for Kimi K2
def execute_cli_tool(tool_name: str, command: str, arguments: Dict[str, Any]) -> str:
    """Execute a CLI tool with given arguments and safety checks"""
    try:
        # Build command with arguments
        if arguments:
            # Convert arguments to command line args
            args = []
            for key, value in arguments.items():
                # Basic argument sanitization
                str_value = str(value)
                if any(char in str_value for char in ["'", '"', "$", "`", "\\"]):
                    return f"Error: Invalid characters in arguments"
                args.append(str_value)
            full_command = f"{command} {' '.join(args)}"
        else:
            full_command = command
        
        # Safety check
        is_safe, safety_msg = is_command_safe(full_command)
        if not is_safe:
            return f"Safety Error: {safety_msg}"
        
        # Log command execution (basic audit trail)
        if os.getenv("LOG_COMMANDS", "").lower() == "true":
            with open("command_audit.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - Tool: {tool_name} - Command: {full_command}\n")
        
        # Execute command with restrictions
        result = subprocess.run(
            full_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
            env={**os.environ, "PATH": "/usr/bin:/bin:/usr/local/bin"}  # Restricted PATH
        )
        
        if result.returncode == 0:
            return result.stdout.strip() if result.stdout else "(No output)"
        else:
            return f"Error: {result.stderr.strip() if result.stderr else f'Exit code {result.returncode}'}"
            
    except subprocess.TimeoutExpired:
        return f"Error: Tool '{tool_name}' timed out after 30 seconds"
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

# Kimi conversation session management
KIMI_SESSIONS_DIR = Path(".kimi_sessions")
# Create directory only when needed, not at module load
# KIMI_SESSIONS_DIR.mkdir(exist_ok=True)

def save_session(session_id: str, messages: List[Dict], metadata: Dict = None) -> None:
    """Save a conversation session to disk"""
    KIMI_SESSIONS_DIR.mkdir(exist_ok=True)  # Ensure directory exists
    session_file = KIMI_SESSIONS_DIR / f"{session_id}.json"
    session_data = {
        "session_id": session_id,
        "messages": messages,
        "metadata": metadata or {},
        "created": metadata.get("created", datetime.now().isoformat()),
        "last_accessed": datetime.now().isoformat(),
        "message_count": len(messages),
        "topics": metadata.get("topics", session_id.split("_")[0].split("-"))  # Extract topics from ID
    }
    
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

def load_session(session_id: str) -> Optional[Dict]:
    """Load a conversation session from disk"""
    if session_id == "@last":
        # Find the most recently accessed session
        sessions = list(KIMI_SESSIONS_DIR.glob("*.json"))
        if not sessions:
            return None
        latest = max(sessions, key=lambda p: p.stat().st_mtime)
        session_id = latest.stem
    
    session_file = KIMI_SESSIONS_DIR / f"{session_id}.json"
    if not session_file.exists():
        return None
    
    with open(session_file, "r") as f:
        return json.load(f)

def list_sessions() -> List[Dict]:
    """List all available sessions"""
    sessions = []
    if not KIMI_SESSIONS_DIR.exists():
        return sessions
    for session_file in KIMI_SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
                sessions.append({
                    "session_id": data["session_id"],
                    "created": data.get("created", "Unknown"),
                    "last_accessed": data.get("last_accessed", "Unknown"),
                    "message_count": data.get("message_count", 0)
                })
        except:
            continue
    
    return sorted(sessions, key=lambda x: x["last_accessed"], reverse=True)

def clear_session(session_id: str) -> bool:
    """Clear a specific session or all sessions"""
    if session_id == "@all":
        for session_file in KIMI_SESSIONS_DIR.glob("*.json"):
            session_file.unlink()
        return True
    else:
        session_file = KIMI_SESSIONS_DIR / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()
            return True
    return False

def truncate_messages_for_context(messages: List[Dict], max_tokens: int = 100000) -> List[Dict]:
    """Truncate old messages if conversation is too long"""
    # Simple truncation - in production you'd count actual tokens
    # Keep system message and recent messages
    if not messages:
        return messages
    
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    
    # Keep last N messages (rough approximation)
    max_messages = 50  # Adjust based on typical message length
    if len(other_msgs) > max_messages:
        other_msgs = other_msgs[-max_messages:]
    
    return system_msgs + other_msgs

# Built-in tool definitions (examples)
BUILTIN_TOOLS = {
    "get_current_time": {
        "command": "date '+%Y-%m-%d %H:%M:%S'",
        "schema": {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current date and time",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    },
    "add_numbers": {
        "command": "python3 -c 'import sys; print(float(sys.argv[1]) + float(sys.argv[2]))'",
        "schema": {
            "type": "function",
            "function": {
                "name": "add_numbers",
                "description": "Add two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"}
                    },
                    "required": ["a", "b"]
                }
            }
        }
    },
    "list_files": {
        "command": "ls -la",
        "schema": {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in current directory",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
    }
}

# Initialize server
server = Server("llm-tools")

# Cost tracking (prices per 1M tokens)
COST_PER_1M_TOKENS = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},   # For <200k tokens
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},  # Actual pricing
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},      # Actual pricing  
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},      # Actual pricing
    "kimi-k2-0711-preview": {"input": 0.60, "output": 2.50},  # Official Kimi K2 pricing
    "moonshot-v1-auto": {"input": 0.15, "output": 2.50},  # Auto-routing model
    "moonshot-v1-128k": {"input": 0.60, "output": 2.50},  # 128k context model
}

# Track costs in memory (reset on server restart)
cost_tracker = defaultdict(lambda: {"requests": 0, "total_cost": 0.0, "tokens": {"input": 0, "output": 0}})

# Cost log file (optional)
COST_LOG_FILE = os.getenv("LLM_COST_LOG", None)

# Initialize LLM clients
anthropic_client = None
openai_client = None
moonshot_client = None
gemini_pro_model = None
gemini_flash_model = None

def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 characters)"""
    return len(text) // 4

def calculate_cost(model: str, input_text: str, output_text: str) -> float:
    """Calculate cost for a request"""
    if model not in COST_PER_1M_TOKENS:
        return 0.0
    
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    
    input_cost = (input_tokens / 1_000_000) * COST_PER_1M_TOKENS[model]["input"]
    output_cost = (output_tokens / 1_000_000) * COST_PER_1M_TOKENS[model]["output"]
    
    return input_cost + output_cost

def track_cost(model: str, input_text: str, output_text: str) -> Dict[str, Any]:
    """Track cost and return cost info"""
    cost = calculate_cost(model, input_text, output_text)
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    
    # Update tracker
    cost_tracker[model]["requests"] += 1
    cost_tracker[model]["total_cost"] += cost
    cost_tracker[model]["tokens"]["input"] += input_tokens
    cost_tracker[model]["tokens"]["output"] += output_tokens
    
    cost_info = {
        "model": model,
        "cost": round(cost, 6),
        "tokens": {"input": input_tokens, "output": output_tokens},
        "cumulative": {
            "total_cost": round(cost_tracker[model]["total_cost"], 4),
            "requests": cost_tracker[model]["requests"]
        }
    }
    
    # Log to file if configured
    if COST_LOG_FILE:
        try:
            with open(COST_LOG_FILE, 'a') as f:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "model": model,
                    "cost": cost_info["cost"],
                    "tokens": cost_info["tokens"],
                    "cumulative_cost": cost_info["cumulative"]["total_cost"]
                }
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            pass  # Silent fail on logging
    
    return cost_info

# Initialize unified session manager if enabled
unified_session_manager = None
model_adapters = {}

def initialize_unified_sessions():
    """Initialize unified session management"""
    global unified_session_manager, model_adapters
    
    unified_session_manager = SessionManager()
    
    # Initialize adapters
    if os.getenv("GOOGLE_API_KEY"):
        model_adapters["gemini"] = GeminiAdapter(api_key=os.getenv("GOOGLE_API_KEY"))
    if os.getenv("OPENAI_API_KEY"):
        model_adapters["openai"] = OpenAIAdapter(api_key=os.getenv("OPENAI_API_KEY"))
    if os.getenv("MOONSHOT_API_KEY"):
        model_adapters["kimi"] = KimiAdapter(api_key=os.getenv("MOONSHOT_API_KEY"))
    if os.getenv("ANTHROPIC_API_KEY"):
        model_adapters["anthropic"] = AnthropicAdapter(api_key=os.getenv("ANTHROPIC_API_KEY"))

def get_adapter_for_model(model: str) -> Optional[Any]:
    """Get the appropriate adapter for a model"""
    # Map model to adapter
    model_lower = model.lower()
    
    if "gemini" in model_lower:
        return model_adapters.get("gemini")
    elif "gpt" in model_lower or "o1" in model_lower:
        return model_adapters.get("openai")
    elif "kimi" in model_lower or "moonshot" in model_lower:
        return model_adapters.get("kimi")
    elif "claude" in model_lower:
        return model_adapters.get("anthropic")
    
    return None

async def _chat_unified(
    messages: List[Dict],
    model: str,
    session_id: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    **kwargs
) -> Tuple[str, Dict]:
    """
    Unified chat implementation that all model-specific functions use.
    
    Args:
        messages: List of message dictionaries
        model: Model identifier
        session_id: Optional session ID for persistence
        temperature: Temperature for sampling
        max_tokens: Maximum output tokens
        stream: Whether to stream the response
        **kwargs: Additional model-specific parameters
        
    Returns:
        Tuple of (response_content, metadata)
    """
    # Get the appropriate adapter
    adapter = get_adapter_for_model(model)
    if not adapter:
        raise ValueError(f"No adapter found for model: {model}")
    
    # Initialize adapter if needed
    if not adapter._client:
        adapter.initialize_client()
    
    # Load or create session
    session = None
    if session_id:
        session = unified_session_manager.load_session(session_id)
        if not session:
            # Create new session
            system_prompt = None
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                    break
            
            session = unified_session_manager.create_session(
                model=model,
                system_prompt=system_prompt,
                session_id=session_id
            )
        else:
            # Check for model switch
            if session["model"] != model:
                unified_session_manager.switch_model(
                    session_id,
                    model,
                    reason="User requested different model"
                )
    
    # Add new messages to session
    if session:
        # Just add all messages from the current request
        # They are always new since 'messages' only contains the current request
        for msg in messages:
            unified_session_manager.add_message(
                session_id,
                msg["role"],
                msg["content"],
                metadata=msg.get("metadata", {})
            )
        
        # Reload session to get the updated messages
        session = unified_session_manager.load_session(session_id)
    
    # Use full session history if available, otherwise just the new messages
    messages_to_send = session["messages"] if session else messages
    
    # Debug: Print what we're sending
    import sys
    print(f"\n=== DEBUG: _chat_unified ===", file=sys.stderr)
    print(f"Session ID: {session_id}", file=sys.stderr)
    print(f"Model: {model}", file=sys.stderr)
    print(f"Session exists: {session is not None}", file=sys.stderr)
    if session:
        print(f"Session messages count: {len(session['messages'])}", file=sys.stderr)
    print(f"Messages to send count: {len(messages_to_send)}", file=sys.stderr)
    for i, msg in enumerate(messages_to_send[:5]):  # First 5 messages
        print(f"  Msg {i}: {msg.get('role', 'unknown')}: {msg.get('content', '')[:50]}...", file=sys.stderr)
    print("=== END DEBUG ===\n", file=sys.stderr)
    
    # Create completion
    response_content, metadata = adapter.create_completion(
        messages_to_send,
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs
    )
    
    # Handle streaming response
    if stream:
        # For streaming, we need to accumulate the content
        # This is a generator, so we need to handle it differently
        return response_content, metadata
    
    # Add response to session
    if session:
        unified_session_manager.add_message(
            session_id,
            "assistant",
            response_content,
            metadata=metadata
        )
    
    # Track costs if available
    if "cost" in metadata:
        cost_tracker[model]["requests"] += 1
        cost_tracker[model]["total_cost"] += metadata["cost"]
        if "tokens" in metadata:
            cost_tracker[model]["tokens"]["input"] += metadata["tokens"].get("input", 0)
            cost_tracker[model]["tokens"]["output"] += metadata["tokens"].get("output", 0)
    
    return response_content, metadata

def init_clients():
    """Initialize LLM clients with API keys"""
    global anthropic_client, openai_client, moonshot_client, gemini_pro_model, gemini_flash_model
    
    # Initialize unified sessions if enabled
    initialize_unified_sessions()
    
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    if os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if os.getenv("MOONSHOT_API_KEY"):
        # Kimi K2 uses OpenAI-compatible API
        moonshot_client = OpenAI(
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.ai/v1"
        )
    
    if os.getenv("GOOGLE_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        gemini_pro_model = genai.GenerativeModel('gemini-2.5-pro')
        gemini_flash_model = genai.GenerativeModel('gemini-2.5-flash')

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available LLM tools"""
    return [
        types.Tool(
            name="analyze_with_gemini",
            description="Analyze large codebases or documents with Gemini 2.5 Pro's massive context window",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The analysis query. Supports file interpolation: {file:path/to/file.py}, {file:path:10-20}, {files:*.py}"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths or glob patterns to include"
                    },
                    "context": {
                        "type": "string",
                        "description": "Direct context string (alternative to files)"
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for conversation persistence"
                    }
                },
                "required": ["prompt"]
            }
        ),
        types.Tool(
            name="quick_gpt",
            description="Fast responses using GPT-4.1-nano for simple tasks",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task or question. Supports file interpolation: {file:path/to/file.py}, {file:path:10-20}, {files:*.py}"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Control randomness (0-1)",
                        "default": 0.3
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for conversation persistence"
                    }
                },
                "required": ["prompt"]
            }
        ),
        types.Tool(
            name="balanced_llm",
            description="Use Gemini 2.5 Flash or GPT-4.1-mini for balanced tasks",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task or question. Supports file interpolation: {file:path/to/file.py}, {file:path:10-20}, {files:*.py}"
                    },
                    "model": {
                        "type": "string",
                        "enum": ["gemini-flash", "gpt-mini"],
                        "description": "Choose model: gemini-flash or gpt-mini",
                        "default": "gemini-flash"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum response length",
                        "default": 1000
                    }
                },
                "required": ["prompt"]
            }
        ),
        types.Tool(
            name="route_to_best_model",
            description="Automatically choose the best model based on the task",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The task description. Supports file interpolation: {file:path/to/file.py}, {file:path:10-20}, {files:*.py}"
                    },
                    "task_type": {
                        "type": "string",
                        "enum": ["analysis", "generation", "simple", "auto"],
                        "description": "Type of task to help routing",
                        "default": "auto"
                    }
                },
                "required": ["prompt"]
            }
        ),
        types.Tool(
            name="kimi_chat",
            description="Flexible Kimi chat - supports simple prompts, multi-turn conversations, and partial pre-filling",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Simple prompt (for single-turn) or latest user message. Supports file interpolation: {file:path/to/file.py}, {file:path:10-20}, {files:*.py}"
                    },
                    "messages": {
                        "type": "array",
                        "description": "Full conversation history (overrides prompt if provided)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["system", "user", "assistant"]},
                                "content": {"type": "string"},
                                "name": {"type": "string"},
                                "partial": {"type": "boolean"}
                            },
                            "required": ["role", "content"]
                        }
                    },
                    "system": {
                        "type": "string",
                        "description": "System message (used only if messages not provided)"
                    },
                    "partial_response": {
                        "type": "string",
                        "description": "Pre-fill the assistant's response to maintain character/format"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model name: 'kimi-k2-0711-preview' (instruct/chat), 'kimi-k2-base' (base/completions), or other Moonshot models",
                        "default": "kimi-k2-0711-preview"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Control randomness (0-1)",
                        "default": 0.6
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum response length",
                        "default": 4096
                    },
                    "available_tools": {
                        "type": "array",
                        "description": "List of tool names that Kimi can execute",
                        "items": {"type": "string"}
                    },
                    "dynamic_tools": {
                        "type": "array",
                        "description": "Dynamic tool definitions created on-the-fly",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "command": {"type": "string"},
                                "schema": {"type": "object"}
                            },
                            "required": ["name", "command", "schema"]
                        }
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for persistent conversations. Use '@last' to continue most recent, '@list' to list sessions, '@clear:id' to clear. For new sessions, use format: 'keyword1-keyword2-keyword3' (will auto-append timestamp)"
                    },
                    "return_conversation": {
                        "type": "boolean",
                        "description": "Return the full conversation history along with the response",
                        "default": False
                    }
                },
                "required": []
            }
        ),
        types.Tool(
            name="check_costs",
            description="Check cumulative costs for all LLM usage",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution"""
    
    if name == "analyze_with_gemini":
        prompt = arguments["prompt"]
        files = arguments.get("files", [])
        context = arguments.get("context", "")
        session_id = arguments.get("session_id", None)
        
        # Process file interpolation in prompt
        if "{file:" in prompt or "{files:" in prompt:
            prompt = construct_prompt(prompt)
        
        # Load files if provided
        if files:
            file_contents = []
            for pattern in files:
                for filepath in glob.glob(pattern, recursive=True):
                    if Path(filepath).is_file():
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                                file_contents.append(f"=== {filepath} ===\n{content}\n")
                        except Exception as e:
                            file_contents.append(f"=== {filepath} ===\nError reading: {e}\n")
            
            context = "\n".join(file_contents) + "\n" + context
        
        # Prepare prompt with context
        full_prompt = f"Context:\n{context}\n\nQuery: {prompt}" if context else prompt
        
        try:
            # Use unified chat
            messages = [{"role": "user", "content": full_prompt}]
            response_content, metadata = await _chat_unified(
                messages=messages,
                model="gemini-2.5-pro",
                session_id=session_id,
                max_tokens=8192
            )
            
            # Format response with cost info
            cost_str = ""
            if "cost" in metadata:
                cost_str = f"\n\n---\n💰 Cost: ${metadata['cost']:.6f}"
                if cost_tracker.get("gemini-2.5-pro"):
                    cost_str += f" | Total: ${cost_tracker['gemini-2.5-pro']['total_cost']:.4f}"
            
            return [types.TextContent(
                type="text",
                text=f"{response_content}{cost_str}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Gemini error: {str(e)}")]
    
    elif name == "quick_gpt":
        prompt = arguments["prompt"]
        temperature = arguments.get("temperature", 0.3)
        session_id = arguments.get("session_id", None)
        
        # Process file interpolation in prompt
        if "{file:" in prompt or "{files:" in prompt:
            prompt = construct_prompt(prompt)
        
        try:
            # Use unified chat
            messages = [{"role": "user", "content": prompt}]
            response_content, metadata = await _chat_unified(
                messages=messages,
                model="gpt-4.1-nano",
                session_id=session_id,
                temperature=temperature,
                max_tokens=500  # Keep responses quick
            )
            
            # Format response with cost info
            cost_str = ""
            if "cost" in metadata:
                cost_str = f"\n\n---\n💰 Cost: ${metadata['cost']:.6f}"
                if cost_tracker.get("gpt-4.1-nano"):
                    cost_str += f" | Total: ${cost_tracker['gpt-4.1-nano']['total_cost']:.4f}"
            
            return [types.TextContent(
                type="text",
                text=f"{response_content}{cost_str}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"OpenAI error: {str(e)}")]
    
    elif name == "balanced_llm":
        prompt = arguments["prompt"]
        model_choice = arguments.get("model", "gemini-flash")
        max_tokens = arguments.get("max_tokens", 1000)
        
        # Process file interpolation in prompt
        if "{file:" in prompt or "{files:" in prompt:
            prompt = construct_prompt(prompt)
        
        if model_choice == "gemini-flash":
            if not gemini_flash_model:
                return [types.TextContent(
                    type="text",
                    text="Error: Gemini API key not configured. Set GOOGLE_API_KEY in .env"
                )]
            
            try:
                response = gemini_flash_model.generate_content(prompt)
                
                # Track cost
                cost_info = track_cost("gemini-2.5-flash", prompt, response.text)
                
                return [types.TextContent(
                    type="text",
                    text=f"{response.text}\n\n---\n💰 Cost: ${cost_info['cost']:.6f} | Total: ${cost_info['cumulative']['total_cost']:.4f}"
                )]
            except Exception as e:
                return [types.TextContent(type="text", text=f"Gemini Flash error: {str(e)}")]
        
        else:  # gpt-mini
            if not openai_client:
                return [types.TextContent(
                    type="text",
                    text="Error: OpenAI API key not configured. Set OPENAI_API_KEY in .env"
                )]
            
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens
                )
                
                # Track cost
                output_text = response.choices[0].message.content
                cost_info = track_cost("gpt-4.1-mini", prompt, output_text)
                
                return [types.TextContent(
                    type="text",
                    text=f"{output_text}\n\n---\n💰 Cost: ${cost_info['cost']:.6f} | Total: ${cost_info['cumulative']['total_cost']:.4f}"
                )]
            except Exception as e:
                return [types.TextContent(type="text", text=f"GPT-4.1-mini error: {str(e)}")]
    
    elif name == "route_to_best_model":
        prompt = arguments["prompt"]
        task_type = arguments.get("task_type", "auto")
        
        # Simple routing logic
        if task_type == "analysis" or len(prompt) > 5000:
            # Large context or analysis -> Gemini
            return await handle_call_tool("analyze_with_gemini", {"prompt": prompt})
        elif task_type == "simple" or len(prompt) < 200:
            # Simple task -> GPT-4.1-nano
            return await handle_call_tool("quick_gpt", {"prompt": prompt})
        else:
            # Default to Gemini Flash for balanced tasks
            return await handle_call_tool("balanced_llm", {"prompt": prompt, "model": "gemini-flash"})
    
    elif name == "kimi_chat":
        if not moonshot_client:
            return [types.TextContent(
                type="text",
                text="Error: Moonshot API key not configured. Set MOONSHOT_API_KEY in .env"
            )]
        
        # Extract parameters
        messages = arguments.get("messages", None)
        prompt = arguments.get("prompt", "")
        system = arguments.get("system", None)
        
        # Process file interpolation in prompt if provided
        if prompt and ("{file:" in prompt or "{files:" in prompt):
            prompt = construct_prompt(prompt)
        partial_response = arguments.get("partial_response", None)
        model = arguments.get("model", "kimi-k2-0711-preview")
        temperature = arguments.get("temperature", 0.6)
        max_tokens = arguments.get("max_tokens", 4096)
        available_tools = arguments.get("available_tools", None)
        dynamic_tools = arguments.get("dynamic_tools", None)
        session_id = arguments.get("session_id", None)
        return_conversation = arguments.get("return_conversation", False)
        
        # Handle special session commands
        if session_id and session_id.startswith("@"):
            if session_id == "@list":
                sessions = list_sessions()
                if not sessions:
                    return [types.TextContent(type="text", text="No active sessions found.")]
                
                report = "## Active Kimi Sessions\n\n"
                for sess in sessions:
                    # Parse session ID to extract base topics
                    base_topics = sess['session_id'].split('_')[0].replace('-', ' ')
                    created_date = sess['created'].split('T')[0] if 'T' in sess['created'] else sess['created']
                    last_date = sess['last_accessed'].split('T')[0] if 'T' in sess['last_accessed'] else sess['last_accessed']
                    
                    report += f"- **{sess['session_id']}**\n"
                    report += f"  - Topics: {base_topics}\n"
                    report += f"  - Messages: {sess['message_count']}\n"
                    report += f"  - Created: {created_date}\n"
                    report += f"  - Last accessed: {last_date}\n\n"
                return [types.TextContent(type="text", text=report)]
            
            elif session_id.startswith("@clear:"):
                sess_to_clear = session_id[7:]  # Remove "@clear:"
                if clear_session(sess_to_clear):
                    return [types.TextContent(type="text", text=f"Session '{sess_to_clear}' cleared successfully.")]
                else:
                    return [types.TextContent(type="text", text=f"Session '{sess_to_clear}' not found.")]
        
        # Load session if specified
        session_metadata = {}
        actual_session_id = None  # Initialize for all code paths
        if session_id and not session_id.startswith("@clear"):
            # First try to load the session (handles @last)
            session_data = load_session(session_id)
            
            if session_data:
                # Session exists (or @last resolved to existing session)
                actual_session_id = session_data.get("session_id", session_id)
            else:
                # New session - check if it needs timestamp
                if not any(session_id.endswith(f"_{i}") for i in range(10)) and "_" not in session_id[-5:]:
                    # This looks like a new session with keywords, add timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    actual_session_id = f"{session_id}_{timestamp}"
                else:
                    # Already has timestamp
                    actual_session_id = session_id
            
            if session_data:
                # Continue existing session
                final_messages = session_data["messages"].copy()
                session_metadata = session_data.get("metadata", {})
                
                # Add new prompt if provided
                if prompt:
                    final_messages.append({"role": "user", "content": prompt})
            else:
                # Start new session
                final_messages = []
                if system:
                    final_messages.append({"role": "system", "content": system})
                if prompt:
                    final_messages.append({"role": "user", "content": prompt})
                session_metadata["created"] = datetime.now().isoformat()
                
                # Extract topics from session ID
                if actual_session_id:
                    base_id = actual_session_id.split("_")[0]
                    session_metadata["topics"] = base_id.split("-")
        else:
            # Build messages from inputs (no session)
            if messages:
                # Use provided message history
                final_messages = messages.copy()
            else:
                # Build from simple inputs
                final_messages = []
                if system:
                    final_messages.append({"role": "system", "content": system})
                if prompt:
                    final_messages.append({"role": "user", "content": prompt})
        
        # Add partial pre-filling if requested
        if partial_response:
            final_messages.append({
                "role": "assistant",
                "content": partial_response,
                "partial": True
            })
        
        # Validate we have messages
        if not final_messages:
            return [types.TextContent(
                type="text",
                text="Error: Must provide either 'prompt' or 'messages'"
            )]
        
        # Prepare tools if available
        tools = None
        all_tools = {}  # Combined tool registry for this request
        
        # Add built-in tools if requested
        if available_tools:
            for tool_name in available_tools:
                if tool_name in BUILTIN_TOOLS:
                    all_tools[tool_name] = BUILTIN_TOOLS[tool_name]
                else:
                    return [types.TextContent(
                        type="text",
                        text=f"Error: Unknown tool '{tool_name}'. Available tools: {list(BUILTIN_TOOLS.keys())}"
                    )]
        
        # Add dynamic tools if provided
        if dynamic_tools:
            for tool_def in dynamic_tools:
                tool_name = tool_def["name"]
                all_tools[tool_name] = {
                    "command": tool_def["command"],
                    "schema": tool_def["schema"]
                }
        
        # Build tools array for API call
        if all_tools:
            tools = [tool_info["schema"] for tool_info in all_tools.values()]
        
        try:
            # Make initial API call
            api_params = {
                "model": model,
                "messages": final_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if tools:
                api_params["tools"] = tools
            
            response = moonshot_client.chat.completions.create(**api_params)
            message = response.choices[0].message
            
            # Check if Kimi wants to call tools
            if hasattr(message, 'tool_calls') and message.tool_calls:
                # Handle tool calls
                tool_results = []
                
                # Add assistant's message with tool calls to conversation
                final_messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args_str = tool_call.function.arguments
                    
                    # Parse arguments
                    try:
                        tool_args = json.loads(tool_args_str) if tool_args_str else {}
                    except json.JSONDecodeError:
                        tool_args = {}
                    
                    # Execute tool
                    if tool_name in all_tools:
                        command = all_tools[tool_name]["command"]
                        result = execute_cli_tool(tool_name, command, tool_args)
                    else:
                        result = f"Error: Unknown tool '{tool_name}'"
                    
                    # Add tool result to conversation
                    final_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                    
                    tool_results.append(f"🔧 {tool_name}({tool_args_str}) → {result}")
                
                # Make follow-up API call with tool results
                follow_up_response = moonshot_client.chat.completions.create(
                    model=model,
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                output_text = follow_up_response.choices[0].message.content
                
                # Include tool execution details
                tool_summary = "\n".join(tool_results)
                final_output = f"{output_text}\n\n**Tool Executions:**\n{tool_summary}"
                
            else:
                # Regular text response
                output_text = message.content
                
                # If partial was used, prepend it to output for display
                if partial_response:
                    output_text = partial_response + output_text
                
                final_output = output_text
            
            # Add assistant's response to conversation
            if final_output:
                # For tool responses, we already added the assistant message
                if not (hasattr(message, 'tool_calls') and message.tool_calls):
                    final_messages.append({"role": "assistant", "content": final_output})
            
            # Save session if specified
            if actual_session_id or (session_id and not session_id.startswith("@")):
                # Use actual_session_id if available (for new sessions with timestamp)
                save_id = actual_session_id or session_id
                # Truncate if too long before saving
                truncated_messages = truncate_messages_for_context(final_messages)
                save_session(save_id, truncated_messages, session_metadata)
            
            # Track cost - estimate input from all messages
            input_text = " ".join([m.get("content", "") for m in final_messages if isinstance(m.get("content"), str)])
            cost_info = track_cost(model, input_text, final_output)
            
            # Prepare response
            response_text = f"{final_output}\n\n---\n💰 Cost: ${cost_info['cost']:.6f} | Total: ${cost_info['cumulative']['total_cost']:.4f}"
            
            # Return conversation if requested
            if return_conversation:
                return [
                    types.TextContent(type="text", text=response_text),
                    types.TextContent(
                        type="text", 
                        text=f"\n\n---\n🗨️ Conversation State:\n```json\n{json.dumps(final_messages, indent=2)}\n```"
                    )
                ]
            else:
                return [types.TextContent(type="text", text=response_text)]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Kimi Chat error: {str(e)}")]
    
    elif name == "check_costs":
        # Generate cost report
        report = "## LLM Usage Cost Report\n\n"
        
        total_cost = 0
        total_requests = 0
        
        for model, data in sorted(cost_tracker.items()):
            if data["requests"] > 0:
                report += f"### {model}\n"
                report += f"- Requests: {data['requests']}\n"
                report += f"- Total Cost: ${data['total_cost']:.4f}\n"
                report += f"- Input Tokens: {data['tokens']['input']:,}\n"
                report += f"- Output Tokens: {data['tokens']['output']:,}\n"
                report += f"- Avg Cost/Request: ${data['total_cost']/data['requests']:.6f}\n\n"
                
                total_cost += data["total_cost"]
                total_requests += data["requests"]
        
        if total_requests == 0:
            report = "No LLM usage tracked yet."
        else:
            report += f"### Total\n"
            report += f"- All Requests: {total_requests}\n"
            report += f"- **Total Cost: ${total_cost:.4f}**\n"
        
        return [types.TextContent(type="text", text=report)]
    
    else:
        return [types.TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]

async def main():
    """Run the MCP server"""
    # Initialize clients
    init_clients()
    
    # Run server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="llm-tools",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())