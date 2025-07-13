"""Base adapter class for model-specific implementations"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import time


class BaseAdapter(ABC):
    """Abstract base class for model adapters"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the adapter
        
        Args:
            api_key: API key for the model provider
        """
        self.api_key = api_key
        self._client = None
    
    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """List of model names this adapter supports"""
        pass
    
    @abstractmethod
    def initialize_client(self):
        """Initialize the API client"""
        pass
    
    @abstractmethod
    def format_messages(self, messages: List[Dict]) -> Any:
        """Format messages for the specific API
        
        Args:
            messages: List of message dictionaries with role/content
            
        Returns:
            Formatted messages in the provider's expected format
        """
        pass
    
    @abstractmethod
    def create_completion(
        self,
        messages: List[Dict],
        model: str,
        **kwargs
    ) -> Tuple[str, Dict]:
        """Create a completion using the model
        
        Args:
            messages: List of message dictionaries
            model: Model identifier
            **kwargs: Additional model-specific parameters
            
        Returns:
            Tuple of (response_content, metadata)
        """
        pass
    
    def extract_metadata(self, response: Any) -> Dict:
        """Extract metadata from API response
        
        Args:
            response: Raw API response
            
        Returns:
            Dictionary containing metadata like tokens, cost, etc.
        """
        return {
            "timestamp": time.time(),
            "tokens": 0,
            "cost": 0.0
        }
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for the API call
        
        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        return 0.0
    
    def handle_error(self, error: Exception) -> Dict:
        """Handle API errors in a consistent way
        
        Args:
            error: The exception that occurred
            
        Returns:
            Error information dictionary
        """
        return {
            "error": True,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": time.time()
        }
    
    def supports_model(self, model: str) -> bool:
        """Check if this adapter supports a given model
        
        Args:
            model: Model identifier
            
        Returns:
            True if the model is supported
        """
        return model in self.supported_models
    
    def validate_messages(self, messages: List[Dict]) -> bool:
        """Validate message format
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            True if messages are valid
        """
        if not messages:
            return False
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["system", "user", "assistant", "tool"]:
                return False
        
        return True
    
    def get_token_limits(self, model: str) -> Dict[str, int]:
        """Get token limits for a model
        
        Args:
            model: Model identifier
            
        Returns:
            Dictionary with context_window and max_output limits
        """
        return {
            "context_window": 4096,
            "max_output": 4096
        }