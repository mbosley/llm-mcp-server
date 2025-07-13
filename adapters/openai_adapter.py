"""OpenAI model adapter"""

import os
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI

from .base_adapter import BaseAdapter


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI models"""
    
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "o1-preview": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
    }
    
    # Token limits
    TOKEN_LIMITS = {
        "gpt-4o": {"context_window": 128000, "max_output": 16384},
        "gpt-4o-2024-11-20": {"context_window": 128000, "max_output": 16384},
        "gpt-4o-mini": {"context_window": 128000, "max_output": 16384},
        "gpt-4o-mini-2024-07-18": {"context_window": 128000, "max_output": 16384},
        "gpt-4-turbo": {"context_window": 128000, "max_output": 4096},
        "gpt-4": {"context_window": 8192, "max_output": 4096},
        "gpt-3.5-turbo": {"context_window": 16385, "max_output": 4096},
        "o1-preview": {"context_window": 128000, "max_output": 32768},
        "o1-mini": {"context_window": 128000, "max_output": 65536},
    }
    
    # Model aliases
    MODEL_ALIASES = {
        "gpt-4.1-nano": "gpt-4o-mini",
        "gpt-4.1-mini": "gpt-4o-mini",
    }
    
    @property
    def supported_models(self) -> List[str]:
        """List of supported OpenAI models"""
        return list(self.PRICING.keys()) + list(self.MODEL_ALIASES.keys())
    
    def initialize_client(self):
        """Initialize the OpenAI client"""
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self._client = OpenAI(api_key=self.api_key)
    
    def format_messages(self, messages: List[Dict]) -> List[Dict]:
        """Format messages for OpenAI API
        
        OpenAI expects messages in their standard format
        """
        formatted = []
        
        for msg in messages:
            formatted_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            # Add name if present (for tool messages)
            if "name" in msg:
                formatted_msg["name"] = msg["name"]
            
            # Add tool_calls if present
            if "tool_calls" in msg:
                formatted_msg["tool_calls"] = msg["tool_calls"]
            
            formatted.append(formatted_msg)
        
        return formatted
    
    def create_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Create a completion using OpenAI
        
        Args:
            messages: List of message dictionaries
            model: OpenAI model identifier
            temperature: Temperature for sampling
            max_tokens: Maximum output tokens
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response_content, metadata)
        """
        if not self._client:
            self.initialize_client()
        
        # Resolve model aliases
        actual_model = self.MODEL_ALIASES.get(model, model)
        
        # Format messages
        formatted_messages = self.format_messages(messages)
        
        # Prepare parameters
        params = {
            "model": actual_model,
            "messages": formatted_messages,
            "temperature": temperature,
            "stream": stream
        }
        
        # Add max_tokens if specified
        if max_tokens:
            params["max_tokens"] = max_tokens
        
        # Add other supported parameters
        for key in ["top_p", "n", "stop", "presence_penalty", "frequency_penalty", "logit_bias", "user", "seed", "tools", "tool_choice", "response_format"]:
            if key in kwargs:
                params[key] = kwargs[key]
        
        try:
            # Create completion
            response = self._client.chat.completions.create(**params)
            
            if stream:
                # For streaming, return a generator
                def stream_generator():
                    content = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content += chunk.choices[0].delta.content
                            yield chunk.choices[0].delta.content
                    
                    # Return accumulated content and metadata
                    return content, self.extract_metadata(chunk)
                
                return stream_generator()
            else:
                # Extract content
                content = response.choices[0].message.content
                
                # Extract metadata
                metadata = self.extract_metadata(response)
                metadata["model"] = actual_model
                
                return content, metadata
                
        except Exception as e:
            error_info = self.handle_error(e)
            raise Exception(f"OpenAI API error: {error_info['error_message']}")
    
    def extract_metadata(self, response: Any) -> Dict:
        """Extract metadata from OpenAI response"""
        metadata = super().extract_metadata(response)
        
        # Extract token usage
        if hasattr(response, 'usage'):
            usage = response.usage
            metadata["tokens"] = {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "total": usage.total_tokens
            }
            
            # Calculate cost
            if hasattr(response, 'model'):
                model = response.model
                # Resolve model name variations
                for known_model in self.PRICING:
                    if known_model in model:
                        input_cost = (usage.prompt_tokens / 1_000_000) * self.PRICING[known_model]["input"]
                        output_cost = (usage.completion_tokens / 1_000_000) * self.PRICING[known_model]["output"]
                        metadata["cost"] = round(input_cost + output_cost, 6)
                        break
        
        # Add finish reason
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'finish_reason'):
                metadata["finish_reason"] = choice.finish_reason
        
        # Add system fingerprint if available
        if hasattr(response, 'system_fingerprint'):
            metadata["system_fingerprint"] = response.system_fingerprint
        
        return metadata
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for OpenAI API call"""
        # Resolve aliases
        actual_model = self.MODEL_ALIASES.get(model, model)
        
        if actual_model not in self.PRICING:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * self.PRICING[actual_model]["input"]
        output_cost = (output_tokens / 1_000_000) * self.PRICING[actual_model]["output"]
        
        return round(input_cost + output_cost, 6)
    
    def get_token_limits(self, model: str) -> Dict[str, int]:
        """Get token limits for an OpenAI model"""
        # Resolve aliases
        actual_model = self.MODEL_ALIASES.get(model, model)
        return self.TOKEN_LIMITS.get(actual_model, super().get_token_limits(model))