"""Test fixtures and mock data for comprehensive testing"""

from datetime import datetime
from typing import Dict, List, Any


class MockResponseData:
    """Mock response data for different LLM providers"""
    
    OPENAI_RESPONSE = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21
        }
    }
    
    GEMINI_RESPONSE = {
        "text": "Hello! I'm here to help you with any questions or tasks you have.",
        "usage_metadata": {
            "prompt_token_count": 8,
            "candidates_token_count": 15,
            "total_token_count": 23
        }
    }
    
    KIMI_RESPONSE = {
        "id": "chatcmpl-kimi123",
        "object": "chat.completion", 
        "created": 1677652288,
        "model": "kimi-k2-0711-preview",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "你好！我是Kimi，很高兴为您服务。"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 18,
            "total_tokens": 28
        }
    }
    
    ANTHROPIC_RESPONSE = {
        "id": "msg_claude123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello! I'm Claude, an AI assistant. How can I help you today?"
            }
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 12,
            "output_tokens": 16
        }
    }


class SessionFixtures:
    """Sample session data for testing"""
    
    @staticmethod
    def basic_session(session_id: str = "test_session_123") -> Dict[str, Any]:
        """Create a basic session structure"""
        return {
            "id": session_id,
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "model": "gpt-4o",
            "models_used": ["gpt-4o"],
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "metadata": {
                "system_prompt": "You are a helpful assistant.",
                "tags": [],
                "total_tokens": 0,
                "total_cost": 0.0,
                "model_switches": []
            }
        }
    
    @staticmethod
    def conversation_session(session_id: str = "conv_session_456") -> Dict[str, Any]:
        """Create a session with a conversation"""
        session = SessionFixtures.basic_session(session_id)
        session["messages"].extend([
            {
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": datetime.now().isoformat()
            },
            {
                "role": "assistant", 
                "content": "I'm doing well, thank you! How can I help you today?",
                "timestamp": datetime.now().isoformat(),
                "tokens": {"input": 10, "output": 15, "total": 25}
            },
            {
                "role": "user",
                "content": "Can you help me with Python coding?",
                "timestamp": datetime.now().isoformat()
            }
        ])
        session["metadata"]["total_tokens"] = 25
        return session
    
    @staticmethod
    def multi_model_session(session_id: str = "multi_session_789") -> Dict[str, Any]:
        """Create a session that used multiple models"""
        session = SessionFixtures.conversation_session(session_id)
        session["model"] = "claude-3-5-sonnet-20241022"
        session["models_used"] = ["gpt-4o", "gemini-2.5-pro", "claude-3-5-sonnet-20241022"]
        session["messages"].extend([
            {
                "role": "system",
                "content": "Model switched from gpt-4o to gemini-2.5-pro. Reason: Better for analysis",
                "timestamp": datetime.now().isoformat(),
                "model_switch": True
            },
            {
                "role": "assistant",
                "content": "I can definitely help with Python! What specific aspect are you working on?",
                "timestamp": datetime.now().isoformat(),
                "tokens": {"input": 5, "output": 20, "total": 25}
            },
            {
                "role": "system", 
                "content": "Model switched from gemini-2.5-pro to claude-3-5-sonnet-20241022. Reason: Better for coding",
                "timestamp": datetime.now().isoformat(),
                "model_switch": True
            }
        ])
        session["metadata"]["model_switches"] = [
            {
                "from": "gpt-4o",
                "to": "gemini-2.5-pro", 
                "timestamp": datetime.now().isoformat(),
                "reason": "Better for analysis"
            },
            {
                "from": "gemini-2.5-pro",
                "to": "claude-3-5-sonnet-20241022",
                "timestamp": datetime.now().isoformat(), 
                "reason": "Better for coding"
            }
        ]
        session["metadata"]["total_tokens"] = 50
        return session


class MessageFixtures:
    """Sample message data for testing"""
    
    @staticmethod
    def user_message(content: str = "Hello, AI!") -> Dict[str, Any]:
        """Create a user message"""
        return {
            "role": "user",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def assistant_message(content: str = "Hello! How can I help?", with_tokens: bool = True) -> Dict[str, Any]:
        """Create an assistant message"""
        message = {
            "role": "assistant", 
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        if with_tokens:
            message["tokens"] = {
                "input": len(content) // 4,  # Rough estimate
                "output": len(content) // 4,
                "total": len(content) // 2
            }
        
        return message
    
    @staticmethod
    def system_message(content: str = "You are a helpful assistant.") -> Dict[str, Any]:
        """Create a system message"""
        return {
            "role": "system",
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def tool_call_message() -> Dict[str, Any]:
        """Create a message with tool calls"""
        return {
            "role": "assistant",
            "content": "",
            "timestamp": datetime.now().isoformat(),
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}'
                    }
                }
            ]
        }
    
    @staticmethod
    def tool_response_message() -> Dict[str, Any]:
        """Create a tool response message"""
        return {
            "role": "tool",
            "content": '{"temperature": "72°F", "condition": "sunny"}',
            "timestamp": datetime.now().isoformat(),
            "tool_call_id": "call_123",
            "name": "get_weather"
        }


class AdapterFixtures:
    """Test fixtures for adapter testing"""
    
    @staticmethod
    def get_test_models() -> Dict[str, List[str]]:
        """Get test model lists for each adapter"""
        return {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
            "gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-pro"],
            "kimi": ["kimi-k2-0711-preview", "moonshot-v1-8k", "moonshot-v1-32k"],
            "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]
        }
    
    @staticmethod
    def get_test_messages() -> List[Dict[str, Any]]:
        """Get test message formats"""
        return [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How's the weather?"},
            {
                "role": "assistant", 
                "content": "",
                "tool_calls": [{"id": "123", "function": {"name": "get_weather"}}]
            },
            {
                "role": "tool",
                "content": "Sunny, 75°F", 
                "tool_call_id": "123",
                "name": "get_weather"
            }
        ]