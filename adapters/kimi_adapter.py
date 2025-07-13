"""Kimi/Moonshot model adapter"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI

from .base_adapter import BaseAdapter


class KimiAdapter(BaseAdapter):
    """Adapter for Kimi/Moonshot models"""
    
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "kimi-k2-0711-preview": {"input": 8.00, "output": 8.00},  # ¥8/1M tokens
        "kimi-k2-base": {"input": 8.00, "output": 8.00},
        "moonshot-v1-8k": {"input": 1.50, "output": 1.50},  # ¥12/1M tokens
        "moonshot-v1-32k": {"input": 3.00, "output": 3.00},  # ¥24/1M tokens
        "moonshot-v1-128k": {"input": 7.50, "output": 7.50},  # ¥60/1M tokens
    }
    
    # Token limits
    TOKEN_LIMITS = {
        "kimi-k2-0711-preview": {"context_window": 128000, "max_output": 128000},
        "kimi-k2-base": {"context_window": 1000000, "max_output": 128000},
        "moonshot-v1-8k": {"context_window": 8192, "max_output": 4096},
        "moonshot-v1-32k": {"context_window": 32768, "max_output": 16384},
        "moonshot-v1-128k": {"context_window": 131072, "max_output": 65536},
    }
    
    # API endpoint
    KIMI_API_BASE = "https://api.moonshot.ai/v1"
    
    @property
    def supported_models(self) -> List[str]:
        """List of supported Kimi/Moonshot models"""
        return list(self.PRICING.keys())
    
    def initialize_client(self):
        """Initialize the Kimi client (uses OpenAI-compatible API)"""
        if not self.api_key:
            self.api_key = os.getenv("MOONSHOT_API_KEY")
        
        if not self.api_key:
            raise ValueError("MOONSHOT_API_KEY not found in environment")
        
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.KIMI_API_BASE
        )
    
    def format_messages(self, messages: List[Dict]) -> List[Dict]:
        """Format messages for Kimi API
        
        Kimi uses OpenAI-compatible format but may need special handling
        for tool calls and system messages
        """
        formatted = []
        
        for msg in messages:
            formatted_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            # Handle partial messages for character consistency
            if msg.get("partial", False):
                formatted_msg["partial"] = True
            
            # Add name if present (for tool messages)
            if "name" in msg:
                formatted_msg["name"] = msg["name"]
            
            # Handle tool calls
            if "tool_calls" in msg:
                formatted_msg["tool_calls"] = msg["tool_calls"]
            
            # Handle tool responses
            if msg["role"] == "tool":
                formatted_msg["tool_call_id"] = msg.get("tool_call_id", "")
            
            formatted.append(formatted_msg)
        
        return formatted
    
    def create_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.6,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Create a completion using Kimi
        
        Args:
            messages: List of message dictionaries
            model: Kimi model identifier
            temperature: Temperature for sampling
            max_tokens: Maximum output tokens
            stream: Whether to stream the response
            tools: Tool definitions for function calling
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response_content, metadata)
        """
        if not self._client:
            self.initialize_client()
        
        # Format messages
        formatted_messages = self.format_messages(messages)
        
        # Handle partial response pre-filling
        partial_response = kwargs.pop("partial_response", None)
        if partial_response:
            # Add partial assistant message
            formatted_messages.append({
                "role": "assistant",
                "content": partial_response,
                "partial": True
            })
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": stream
        }
        
        # Add max_tokens if specified
        if max_tokens:
            params["max_tokens"] = max_tokens
        elif model in self.TOKEN_LIMITS:
            # Use model's default max output
            params["max_tokens"] = self.TOKEN_LIMITS[model]["max_output"]
        
        # Add tools if provided
        if tools:
            params["tools"] = tools
            if "tool_choice" in kwargs:
                params["tool_choice"] = kwargs["tool_choice"]
        
        # Add other supported parameters
        for key in ["top_p", "presence_penalty", "frequency_penalty", "stop", "n"]:
            if key in kwargs:
                params[key] = kwargs[key]
        
        try:
            # Create completion
            response = self._client.chat.completions.create(**params)
            
            if stream:
                # For streaming, return a generator
                def stream_generator():
                    content = ""
                    tool_calls = []
                    
                    for chunk in response:
                        # Handle content
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                            yield {"content": chunk.choices[0].delta.content}
                        
                        # Handle tool calls
                        if hasattr(chunk.choices[0].delta, 'tool_calls'):
                            for tc in chunk.choices[0].delta.tool_calls or []:
                                if tc.index >= len(tool_calls):
                                    tool_calls.append({
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                
                                if tc.function.name:
                                    tool_calls[tc.index]["function"]["name"] = tc.function.name
                                if tc.function.arguments:
                                    tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments
                    
                    # Return final content and metadata
                    metadata = self.extract_metadata(chunk)
                    if tool_calls:
                        metadata["tool_calls"] = tool_calls
                    
                    return content, metadata
                
                return stream_generator()
            else:
                # Extract content and tool calls
                choice = response.choices[0]
                content = choice.message.content or ""
                
                # Extract metadata
                metadata = self.extract_metadata(response)
                metadata["model"] = model
                
                # Add tool calls if present
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    metadata["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in choice.message.tool_calls
                    ]
                
                return content, metadata
                
        except Exception as e:
            error_info = self.handle_error(e)
            # Add more context for authentication errors
            if "401" in str(e) or "authentication" in str(e).lower():
                raise Exception(f"Kimi API authentication failed. Please check your MOONSHOT_API_KEY. Error: {error_info['error_message']}")
            raise Exception(f"Kimi API error: {error_info['error_message']}")
    
    def extract_metadata(self, response: Any) -> Dict:
        """Extract metadata from Kimi response"""
        metadata = super().extract_metadata(response)
        
        # Extract token usage
        if hasattr(response, 'usage'):
            usage = response.usage
            metadata["tokens"] = {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens
            }
            
            # Calculate cost (convert from CNY to USD, approximate rate)
            if hasattr(response, 'model'):
                model = response.model
                if model in self.PRICING:
                    # Kimi pricing is in CNY, convert to USD (approximate)
                    cny_to_usd = 0.14  # Approximate conversion rate
                    input_cost = (usage.prompt_tokens / 1_000_000) * self.PRICING[model]["input"] * cny_to_usd
                    output_cost = (usage.completion_tokens / 1_000_000) * self.PRICING[model]["output"] * cny_to_usd
                    metadata["cost"] = round(input_cost + output_cost, 6)
                    metadata["cost_currency"] = "USD (converted from CNY)"
        
        # Add finish reason
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'finish_reason'):
                metadata["finish_reason"] = choice.finish_reason
        
        # Add Kimi-specific metadata if available
        if hasattr(response, 'id'):
            metadata["request_id"] = response.id
        
        return metadata
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Kimi API call (converted to USD)"""
        if model not in self.PRICING:
            return 0.0
        
        # Convert from CNY to USD
        cny_to_usd = 0.14
        input_cost = (input_tokens / 1_000_000) * self.PRICING[model]["input"] * cny_to_usd
        output_cost = (output_tokens / 1_000_000) * self.PRICING[model]["output"] * cny_to_usd
        
        return round(input_cost + output_cost, 6)
    
    def get_token_limits(self, model: str) -> Dict[str, int]:
        """Get token limits for a Kimi model"""
        return self.TOKEN_LIMITS.get(model, super().get_token_limits(model))
    
    def create_dynamic_tool(self, name: str, command: str, schema: Dict) -> Dict:
        """Create a dynamic tool definition for Kimi's tool calling
        
        Args:
            name: Tool name
            command: Command template with placeholders
            schema: JSON schema for parameters
            
        Returns:
            Tool definition compatible with Kimi API
        """
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": schema.get("description", f"Execute {name}"),
                "parameters": schema
            }
        }