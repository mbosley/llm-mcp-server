"""Logging configuration for LLM MCP Server"""

import logging
import sys
import os
from typing import Optional


def setup_logging(name: str = "llm_mcp_server", level: Optional[str] = None) -> logging.Logger:
    """Set up structured logging for the application
    
    Args:
        name: Logger name (defaults to llm_mcp_server)
        level: Logging level (defaults to LLM_LOG_LEVEL env var or INFO)
        
    Returns:
        Configured logger instance
    """
    # Get log level from environment or parameter
    if level is None:
        level = os.getenv('LLM_LOG_LEVEL', 'INFO').upper()
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Don't add handlers if they already exist (avoid duplicates)
    if logger.handlers:
        return logger
    
    # Create console handler
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(getattr(logging, level))
    
    # Create formatter with structured output
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add JSON formatter if requested
    if os.getenv('LLM_LOG_FORMAT', '').lower() == 'json':
        try:
            import json
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_obj = {
                        'timestamp': self.formatTime(record, self.datefmt),
                        'name': record.name,
                        'level': record.levelname,
                        'message': record.getMessage(),
                        'module': record.module,
                        'function': record.funcName,
                        'line': record.lineno
                    }
                    if hasattr(record, 'extra'):
                        log_obj.update(record.extra)
                    return json.dumps(log_obj)
            
            formatter = JSONFormatter()
        except ImportError:
            pass  # Fall back to text formatter
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


# Create default logger instance
logger = setup_logging()