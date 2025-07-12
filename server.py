#!/usr/bin/env python3
"""
LLM MCP Server - Provides access to various LLM APIs as tools
"""

import os
import json
import asyncio
from typing import Optional, List, Dict, Any
from pathlib import Path
import glob
from datetime import datetime
from collections import defaultdict

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# LLM Libraries
import anthropic
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize server
server = Server("llm-tools")

# Cost tracking (prices per 1M tokens)
COST_PER_1M_TOKENS = {
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},   # For <200k tokens
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},  # Actual pricing
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},      # Actual pricing  
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},      # Actual pricing
    "kimi-k2-base": {"input": 0.20, "output": 0.80},      # ~5x cheaper than Claude/Gemini
    "kimi-k2-instruct": {"input": 0.25, "output": 1.00},  # Estimated pricing
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

def init_clients():
    """Initialize LLM clients with API keys"""
    global anthropic_client, openai_client, moonshot_client, gemini_pro_model, gemini_flash_model
    
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    if os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if os.getenv("MOONSHOT_API_KEY"):
        # Kimi K2 uses OpenAI-compatible API
        moonshot_client = OpenAI(
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://platform.moonshot.ai/v1"
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
                        "description": "The analysis query"
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths or glob patterns to include"
                    },
                    "context": {
                        "type": "string",
                        "description": "Direct context string (alternative to files)"
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
                        "description": "The task or question"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Control randomness (0-1)",
                        "default": 0.3
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
                        "description": "The task or question"
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
                        "description": "The task description"
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
            name="kimi_k2_base",
            description="Use Kimi K2 Base model (1T params) for raw completions and experimentation",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The input prompt"
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Control randomness (0-1). Kimi K2 maps as: real_temp = request_temp * 0.6",
                        "default": 0.6
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum response length",
                        "default": 4096
                    }
                },
                "required": ["prompt"]
            }
        ),
        types.Tool(
            name="kimi_k2_instruct",
            description="Use Kimi K2 Instruct model for chat, tool use, and agentic tasks (128k context)",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The user message or task"
                    },
                    "system": {
                        "type": "string",
                        "description": "System message for behavior guidance",
                        "default": "You are Kimi, an AI assistant created by Moonshot AI."
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Control randomness (0-1). Recommended: 0.6",
                        "default": 0.6
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum response length",
                        "default": 4096
                    }
                },
                "required": ["prompt"]
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
        if not gemini_pro_model:
            return [types.TextContent(
                type="text",
                text="Error: Gemini API key not configured. Set GOOGLE_API_KEY in .env"
            )]
        
        prompt = arguments["prompt"]
        files = arguments.get("files", [])
        context = arguments.get("context", "")
        
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
            # Gemini 2.5 Pro has a massive context window
            response = gemini_pro_model.generate_content(full_prompt)
            
            # Track cost
            cost_info = track_cost("gemini-2.5-pro", full_prompt, response.text)
            
            return [types.TextContent(
                type="text", 
                text=f"{response.text}\n\n---\n💰 Cost: ${cost_info['cost']:.6f} | Total: ${cost_info['cumulative']['total_cost']:.4f}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Gemini error: {str(e)}")]
    
    elif name == "quick_gpt":
        if not openai_client:
            return [types.TextContent(
                type="text",
                text="Error: OpenAI API key not configured. Set OPENAI_API_KEY in .env"
            )]
        
        prompt = arguments["prompt"]
        temperature = arguments.get("temperature", 0.3)
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500  # Keep responses quick
            )
            
            # Track cost
            output_text = response.choices[0].message.content
            cost_info = track_cost("gpt-4.1-nano", prompt, output_text)
            
            return [types.TextContent(
                type="text",
                text=f"{output_text}\n\n---\n💰 Cost: ${cost_info['cost']:.6f} | Total: ${cost_info['cumulative']['total_cost']:.4f}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"OpenAI error: {str(e)}")]
    
    elif name == "balanced_llm":
        prompt = arguments["prompt"]
        model_choice = arguments.get("model", "gemini-flash")
        max_tokens = arguments.get("max_tokens", 1000)
        
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
    
    elif name == "kimi_k2_base":
        if not moonshot_client:
            return [types.TextContent(
                type="text",
                text="Error: Moonshot API key not configured. Set MOONSHOT_API_KEY in .env"
            )]
        
        prompt = arguments["prompt"]
        temperature = arguments.get("temperature", 0.6)
        max_tokens = arguments.get("max_tokens", 4096)
        
        try:
            # Use the base model for raw completions
            response = moonshot_client.completions.create(
                model="kimi-k2",
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Track cost
            output_text = response.choices[0].text
            cost_info = track_cost("kimi-k2-base", prompt, output_text)
            
            return [types.TextContent(
                type="text",
                text=f"{output_text}\n\n---\n💰 Cost: ${cost_info['cost']:.6f} | Total: ${cost_info['cumulative']['total_cost']:.4f}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Kimi K2 Base error: {str(e)}")]
    
    elif name == "kimi_k2_instruct":
        if not moonshot_client:
            return [types.TextContent(
                type="text",
                text="Error: Moonshot API key not configured. Set MOONSHOT_API_KEY in .env"
            )]
        
        prompt = arguments["prompt"]
        system = arguments.get("system", "You are Kimi, an AI assistant created by Moonshot AI.")
        temperature = arguments.get("temperature", 0.6)
        max_tokens = arguments.get("max_tokens", 4096)
        
        try:
            # Use chat completions for the instruct model
            response = moonshot_client.chat.completions.create(
                model="kimi-k2-instruct",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Track cost
            output_text = response.choices[0].message.content
            cost_info = track_cost("kimi-k2-instruct", prompt, output_text)
            
            return [types.TextContent(
                type="text",
                text=f"{output_text}\n\n---\n💰 Cost: ${cost_info['cost']:.6f} | Total: ${cost_info['cumulative']['total_cost']:.4f}"
            )]
        except Exception as e:
            return [types.TextContent(type="text", text=f"Kimi K2 Instruct error: {str(e)}")]
    
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