"""Anthropic Claude model adapter"""

import os
from typing import Dict, List, Optional, Any, Tuple
import anthropic

from .base_adapter import BaseAdapter


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude models"""
    
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
        "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    }
    
    # Token limits
    TOKEN_LIMITS = {
        "claude-3-opus-20240229": {"context_window": 200000, "max_output": 4096},
        "claude-3-5-sonnet-20241022": {"context_window": 200000, "max_output": 8192},
        "claude-3-5-sonnet-20240620": {"context_window": 200000, "max_output": 8192},
        "claude-3-5-haiku-20241022": {"context_window": 200000, "max_output": 8192},
        "claude-3-haiku-20240307": {"context_window": 200000, "max_output": 4096},
    }
    
    @property
    def supported_models(self) -> List[str]:
        """List of supported Claude models"""
        return list(self.PRICING.keys())
    
    def initialize_client(self):
        """Initialize the Anthropic client"""
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self._client = anthropic.Anthropic(api_key=self.api_key)
    
    def format_messages(self, messages: List[Dict]) -> Tuple[List[Dict], Optional[str]]:
        """Format messages for Anthropic API
        
        Anthropic has a specific format and requires system message separately
        """
        formatted = []
        system_message = None
        
        for msg in messages:
            if msg["role"] == "system":
                # Anthropic expects system message separately
                system_message = msg["content"]
            else:
                # Convert tool role to user role with special formatting
                if msg["role"] == "tool":
                    formatted.append({
                        "role": "user",
                        "content": f"Tool Response ({msg.get('name', 'unknown')}): {msg['content']}"
                    })
                else:
                    formatted.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        return formatted, system_message
    
    def create_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Create a completion using Claude
        
        Args:
            messages: List of message dictionaries
            model: Claude model identifier
            temperature: Temperature for sampling
            max_tokens: Maximum output tokens
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response_content, metadata)
        """
        if not self._client:
            self.initialize_client()
        
        # Format messages
        formatted_messages, system_message = self.format_messages(messages)
        
        # Prepare parameters
        params = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self.TOKEN_LIMITS[model]["max_output"],
            "stream": stream
        }
        
        # Add system message if present
        if system_message:
            params["system"] = system_message
        
        # Add other supported parameters
        for key in ["top_p", "top_k", "stop_sequences", "metadata"]:
            if key in kwargs:
                params[key] = kwargs[key]
        
        try:
            # Create completion
            response = self._client.messages.create(**params)
            
            if stream:
                # For streaming, return a generator
                def stream_generator():
                    content = ""
                    for event in response:
                        if event.type == "content_block_delta":
                            if hasattr(event.delta, 'text'):
                                content += event.delta.text
                                yield event.delta.text
                        elif event.type == "message_stop":
                            # Extract final metadata
                            metadata = self.extract_metadata(event)
                            return content, metadata
                    
                    return content, {}
                
                return stream_generator()
            else:
                # Extract content
                content = ""
                for block in response.content:
                    if block.type == "text":
                        content += block.text
                
                # Extract metadata
                metadata = self.extract_metadata(response)
                metadata["model"] = model
                
                return content, metadata
                
        except Exception as e:
            error_info = self.handle_error(e)
            raise Exception(f"Anthropic API error: {error_info['error_message']}")
    
    def extract_metadata(self, response: Any) -> Dict:
        """Extract metadata from Anthropic response"""
        metadata = super().extract_metadata(response)
        
        # Extract token usage
        if hasattr(response, 'usage'):
            usage = response.usage
            metadata["tokens"] = {
                "input": usage.input_tokens,
                "output": usage.output_tokens,
                "total": usage.input_tokens + usage.output_tokens
            }
            
            # Calculate cost
            if hasattr(response, 'model'):
                model = response.model
                if model in self.PRICING:
                    input_cost = (usage.input_tokens / 1_000_000) * self.PRICING[model]["input"]
                    output_cost = (usage.output_tokens / 1_000_000) * self.PRICING[model]["output"]
                    metadata["cost"] = round(input_cost + output_cost, 6)
        
        # Add stop reason
        if hasattr(response, 'stop_reason'):
            metadata["finish_reason"] = response.stop_reason
        
        # Add response ID
        if hasattr(response, 'id'):
            metadata["request_id"] = response.id
        
        return metadata
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Anthropic API call"""
        if model not in self.PRICING:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * self.PRICING[model]["input"]
        output_cost = (output_tokens / 1_000_000) * self.PRICING[model]["output"]
        
        return round(input_cost + output_cost, 6)
    
    def get_token_limits(self, model: str) -> Dict[str, int]:
        """Get token limits for a Claude model"""
        return self.TOKEN_LIMITS.get(model, super().get_token_limits(model))