#!/usr/bin/env python3
"""Comprehensive test suite for model adapters"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List

from adapters.base_adapter import BaseAdapter
from adapters.gemini_adapter import GeminiAdapter
from adapters.openai_adapter import OpenAIAdapter
from adapters.kimi_adapter import KimiAdapter
from adapters.anthropic_adapter import AnthropicAdapter


class TestBaseAdapter(unittest.TestCase):
    """Test the base adapter abstract class"""
    
    def test_base_adapter_initialization(self):
        """Test base adapter initialization"""
        # Create a concrete implementation for testing
        class ConcreteAdapter(BaseAdapter):
            @property
            def supported_models(self):
                return ["test-model"]
            
            def initialize_client(self):
                pass
            
            def format_messages(self, messages):
                return messages
            
            def create_completion(self, messages, model, **kwargs):
                return "test response", {"test": "metadata"}
        
        adapter = ConcreteAdapter(api_key="test-key")
        self.assertEqual(adapter.api_key, "test-key")
        self.assertIsNone(adapter._client)


class TestGeminiAdapter(unittest.TestCase):
    """Test Gemini adapter functionality"""
    
    def setUp(self):
        self.adapter = GeminiAdapter(api_key="test-gemini-key")
    
    def test_supported_models(self):
        """Test Gemini supported models list"""
        models = self.adapter.supported_models
        self.assertIn("gemini-2.5-pro", models)
        self.assertIn("gemini-2.5-flash", models)
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
    
    def test_pricing_data(self):
        """Test pricing data structure"""
        for model in self.adapter.supported_models:
            self.assertIn(model, self.adapter.PRICING)
            pricing = self.adapter.PRICING[model]
            self.assertIn("input", pricing)
            self.assertIn("output", pricing)
            self.assertIsInstance(pricing["input"], (int, float))
            self.assertIsInstance(pricing["output"], (int, float))


class TestOpenAIAdapter(unittest.TestCase):
    """Test OpenAI adapter functionality"""
    
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test-openai-key")
    
    def test_supported_models(self):
        """Test OpenAI supported models"""
        models = self.adapter.supported_models
        self.assertIn("gpt-4o", models)
        self.assertIn("gpt-4o-mini", models)
        self.assertIsInstance(models, list)


class TestKimiAdapter(unittest.TestCase):
    """Test Kimi adapter functionality"""
    
    def setUp(self):
        self.adapter = KimiAdapter(api_key="test-kimi-key")
    
    def test_supported_models(self):
        """Test Kimi supported models"""
        models = self.adapter.supported_models
        self.assertIn("kimi-k2-0711-preview", models)
        self.assertIn("moonshot-v1-8k", models)
        self.assertIsInstance(models, list)
    
    def test_api_endpoint(self):
        """Test correct API endpoint is used"""
        self.assertEqual(self.adapter.KIMI_API_BASE, "https://api.moonshot.ai/v1")


class TestAnthropicAdapter(unittest.TestCase):
    """Test Anthropic adapter functionality"""
    
    def setUp(self):
        self.adapter = AnthropicAdapter(api_key="test-anthropic-key")
    
    def test_supported_models(self):
        """Test Anthropic supported models"""
        models = self.adapter.supported_models
        self.assertIn("claude-3-5-sonnet-20241022", models)
        self.assertIn("claude-3-5-haiku-20241022", models)
        self.assertIsInstance(models, list)


if __name__ == "__main__":
    unittest.main()