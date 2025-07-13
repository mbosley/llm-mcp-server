"""Gemini model adapter"""

import os
from typing import Dict, List, Optional, Any, Tuple
import google.generativeai as genai

from .base_adapter import BaseAdapter


class GeminiAdapter(BaseAdapter):
    """Adapter for Google Gemini models"""
    
    # Pricing per 1M tokens (as of 2024)
    PRICING = {
        "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 2.50, "output": 10.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    }
    
    # Token limits
    TOKEN_LIMITS = {
        "gemini-2.5-pro": {"context_window": 2000000, "max_output": 8192},
        "gemini-2.5-flash": {"context_window": 1000000, "max_output": 8192},
        "gemini-2.0-flash": {"context_window": 1000000, "max_output": 8192},
        "gemini-1.5-pro": {"context_window": 2000000, "max_output": 8192},
        "gemini-1.5-flash": {"context_window": 1000000, "max_output": 8192},
    }
    
    @property
    def supported_models(self) -> List[str]:
        """List of supported Gemini models"""
        return list(self.PRICING.keys())
    
    def initialize_client(self):
        """Initialize the Gemini client"""
        if not self.api_key:
            self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        genai.configure(api_key=self.api_key)
    
    def format_messages(self, messages: List[Dict]) -> List[Dict]:
        """Format messages for Gemini API
        
        Gemini expects a specific format for messages
        """
        formatted = []
        
        # Extract system message if present
        system_message = None
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                # Convert role names
                role = "model" if msg["role"] == "assistant" else msg["role"]
                formatted.append({
                    "role": role,
                    "parts": [msg["content"]]
                })
        
        return formatted, system_message
    
    def create_completion(
        self,
        messages: List[Dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Create a completion using Gemini
        
        Args:
            messages: List of message dictionaries
            model: Gemini model identifier
            temperature: Temperature for sampling
            max_tokens: Maximum output tokens
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (response_content, metadata)
        """
        if not self._client:
            self.initialize_client()
        
        # Format messages
        formatted_messages, system_message = self.format_messages(messages)
        
        # Create the model
        genai_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_message
        )
        
        # Set generation config
        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens or self.TOKEN_LIMITS[model]["max_output"],
            **{k: v for k, v in kwargs.items() if k in [
                "top_p", "top_k", "candidate_count", "stop_sequences"
            ]}
        )
        
        try:
            # Generate response
            response = genai_model.generate_content(
                formatted_messages,
                generation_config=generation_config
            )
            
            # Extract content
            content = response.text
            
            # Extract metadata
            metadata = self.extract_metadata(response)
            metadata["model"] = model
            
            return content, metadata
            
        except Exception as e:
            error_info = self.handle_error(e)
            raise Exception(f"Gemini API error: {error_info['error_message']}")
    
    def extract_metadata(self, response: Any) -> Dict:
        """Extract metadata from Gemini response"""
        metadata = super().extract_metadata(response)
        
        # Extract token usage if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            metadata["tokens"] = {
                "input": usage.prompt_token_count,
                "output": usage.candidates_token_count,
                "total": usage.total_token_count
            }
            
            # Calculate cost
            if hasattr(response, 'model_name'):
                model = response.model_name
                if model in self.PRICING:
                    input_cost = (usage.prompt_token_count / 1_000_000) * self.PRICING[model]["input"]
                    output_cost = (usage.candidates_token_count / 1_000_000) * self.PRICING[model]["output"]
                    metadata["cost"] = round(input_cost + output_cost, 6)
        
        # Add finish reason
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                metadata["finish_reason"] = str(candidate.finish_reason)
        
        return metadata
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for Gemini API call"""
        if model not in self.PRICING:
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * self.PRICING[model]["input"]
        output_cost = (output_tokens / 1_000_000) * self.PRICING[model]["output"]
        
        return round(input_cost + output_cost, 6)
    
    def get_token_limits(self, model: str) -> Dict[str, int]:
        """Get token limits for a Gemini model"""
        return self.TOKEN_LIMITS.get(model, super().get_token_limits(model))