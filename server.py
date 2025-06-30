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

# Initialize LLM clients
anthropic_client = None
openai_client = None
gemini_pro_model = None
gemini_flash_model = None

def init_clients():
    """Initialize LLM clients with API keys"""
    global anthropic_client, openai_client, gemini_pro_model, gemini_flash_model
    
    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    if os.getenv("OPENAI_API_KEY"):
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
            return [types.TextContent(type="text", text=response.text)]
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
            return [types.TextContent(type="text", text=response.choices[0].message.content)]
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
                return [types.TextContent(type="text", text=response.text)]
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
                return [types.TextContent(type="text", text=response.choices[0].message.content)]
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